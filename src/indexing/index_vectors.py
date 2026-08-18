import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency may be absent
    def load_dotenv() -> bool:
        return False

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)


class LocalVectorIndex:
    """A small vector index for local retrieval."""

    def __init__(self, dim: int = 768):
        self.dim = dim
        self._vectors: List[List[float]] = []
        self._ids: List[str] = []
        self._payloads: List[Dict[str, Any]] = []

    def add_chunk(self, chunk_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        vector, source = build_embedding(content, dim=self.dim)
        self._ids.append(chunk_id)
        self._vectors.append(vector)
        self._payloads.append({"chunk_id": chunk_id, "content": content, "metadata": metadata or {}, "embedding_source": source})

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._vectors:
            return []
        query_vector, query_source = build_embedding(query, dim=self.dim)
        if query_source == "hash_fallback":
            print("⚠️  Query embedding used the hash fallback, not a real model — results are keyword-ish, not semantic.")
        scored = []
        for chunk_id, vector, payload in zip(self._ids, self._vectors, self._payloads):
            score = cosine_similarity(query_vector, vector)
            scored.append((score, chunk_id, payload))
        scored.sort(key=lambda item: item[0], reverse=True)
        results = []
        for score, chunk_id, payload in scored[:top_k]:
            results.append({
                "chunk_id": chunk_id,
                "score": round(float(score), 6),
                "content": payload["content"],
                "metadata": payload["metadata"],
            })
        return results


def _normalize_vector(vector: List[float], dim: int) -> List[float]:
    if not vector:
        return [0.0] * dim
    normalized = [float(value) for value in vector]
    if len(normalized) >= dim:
        return normalized[:dim]
    return normalized + [0.0] * (dim - len(normalized))


def _try_litellm_embedding(text: str) -> Optional[List[float]]:
    try:
        from litellm import embedding
    except ImportError:
        return None

    model_candidates = []
    explicit_model = os.getenv("EMBEDDING_MODEL") or os.getenv("LITELLM_EMBEDDING_MODEL")
    if explicit_model:
        model_candidates.append(explicit_model)
    model_candidates.extend(["ollama/nomic-embed-text", "nomic-embed-text"])

    for model_name in model_candidates:
        completion_kwargs = {"timeout": 180}
        if model_name.startswith("ollama/"):
            completion_kwargs["api_base"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        try:
            response = embedding(model=model_name, input=[text], **completion_kwargs)
            if hasattr(response, "data"):
                first_item = response.data[0]
                if isinstance(first_item, dict):
                    values = first_item.get("embedding")
                else:
                    values = getattr(first_item, "embedding", None)
            elif isinstance(response, dict):
                values = response.get("data", [{}])[0].get("embedding")
            else:
                values = None
            if values is not None:
                return [float(value) for value in values]
            print(f"⚠️  Embedding call to '{model_name}' returned a response but no usable "
                  f"'embedding' field was found. Raw response type: {type(response)}, "
                  f"repr (truncated): {repr(response)[:300]}")
        except Exception as exc:
            print(f"⚠️  Embedding call to '{model_name}' failed: {exc}")
            continue
    return None


def build_embedding(text: str, dim: int = 768) -> tuple[List[float], str]:
    """Create an embedding vector using LiteLLM/Ollama when available, otherwise
    fall back to a deterministic hash embedding.

    Returns (vector, source) where source is "llm" or "hash_fallback" — callers
    must not discard this, since a hash-fallback vector has no real semantic
    meaning and silently degrades retrieval quality if unnoticed.
    """
    llm_vector = _try_litellm_embedding(text)
    if llm_vector is not None:
        return _normalize_vector(llm_vector, dim), "llm"

    normalized = " ".join(text.lower().split())
    tokens = [token for token in re_split(normalized) if token]
    if not tokens:
        return [0.0] * dim, "hash_fallback"

    vector = [0.0] * dim
    for index, token in enumerate(tokens):
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) / 0xFFFFFFFF
        vector[index % dim] += value
    for i in range(dim):
        vector[i] = round(vector[i], 6)
    return vector, "hash_fallback"


def re_split(text: str) -> List[str]:
    return [part for part in text.replace("\n", " ").replace("/", " ").split() if part]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _try_pgvector_connection(dsn: Optional[str] = None):
    """Connect to Postgres, but do NOT register the vector type here — the
    'vector' extension may not be enabled yet on a fresh database, and
    register_vector() throws if the type doesn't exist. Callers that need
    vector operations must call _ensure_vector_extension() first."""
    if not dsn:
        dsn = os.getenv("PGVECTOR_DSN") or os.getenv("POSTGRES_DSN")
    if not dsn:
        return None
    try:
        import psycopg
    except Exception:
        return None

    try:
        return psycopg.connect(dsn)
    except Exception as exc:
        print(f"⚠️  Could not connect to Postgres: {exc}")
        return None


def _ensure_vector_extension(conn) -> bool:
    """Create the vector extension if missing, then register the type on this
    connection. Returns False (loudly) instead of raising if this fails, so
    callers can fall back rather than crash."""
    try:
        from pgvector.psycopg import register_vector
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        register_vector(conn)
        return True
    except Exception as exc:
        print(f"⚠️  Could not enable/register the pgvector extension: {exc}")
        return False


def get_pgvector_connection(dsn: Optional[str] = None):
    """Public accessor so other modules (e.g. hybrid search) can reuse the same
    connection + extension-registration logic without duplicating it. Returns
    None if Postgres is unreachable OR the vector extension can't be enabled —
    either way, callers should treat this as "fall back", not crash."""
    conn = _try_pgvector_connection(dsn)
    if conn is None:
        return None
    if not _ensure_vector_extension(conn):
        conn.close()
        return None
    return conn


def _store_pgvector_records(records: List[Dict[str, Any]], dsn: Optional[str], table_name: str = "rag_chunks") -> None:
    conn = get_pgvector_connection(dsn)
    if conn is None:
        return
    from psycopg.types.json import Jsonb
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    chunk_id text PRIMARY KEY,
                    embedding vector(768),
                    content text,
                    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
                    metadata jsonb,
                    department text,
                    sensitivity text,
                    created_at text
                )
                """
            )
            # 'simple' config (not 'english') deliberately — this corpus is
            # multilingual (EN/FR/AR), and English stemming rules would
            # mis-stem French/Arabic tokens rather than just leaving them alone.
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {table_name}_hnsw_idx ON {table_name} USING hnsw (embedding vector_cosine_ops)"
            )
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {table_name}_tsv_idx ON {table_name} USING gin (content_tsv)"
            )
            for record in records:
                cur.execute(
                    f"""
                    INSERT INTO {table_name} (chunk_id, embedding, content, metadata, department, sensitivity, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        department = EXCLUDED.department,
                        sensitivity = EXCLUDED.sensitivity,
                        created_at = EXCLUDED.created_at
                    """,
                    (
                        record["chunk_id"],
                        record["embedding"],
                        record["content"],
                        Jsonb(record["metadata"]),
                        record["department"],
                        record["sensitivity"],
                        record["created_at"],
                    ),
                )
            conn.commit()
    finally:
        conn.close()


def _content_hash(text: str) -> str:
    """Same normalization/hash as chunking.chunk_hash() — kept as a local
    fallback so older chunks.jsonl files without a 'content_hash' field
    (produced before that field existed) still get a usable fingerprint
    instead of being treated as permanently uncacheable."""
    normalized = " ".join(text.split()).encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


_DEFAULT_EMBEDDING_CACHE_PATH = Path("data/manifests/embedding_cache.json")


def _load_embedding_cache(cache_path: Path | str = _DEFAULT_EMBEDDING_CACHE_PATH) -> Dict[str, Dict[str, Any]]:
    """Load the persisted chunk_id -> {content_hash, embedding, source}
    cache. Missing/corrupt cache is treated as empty rather than fatal —
    worst case, everything just gets re-embedded once, same as before
    incremental indexing existed."""
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_embedding_cache(cache: Dict[str, Dict[str, Any]], cache_path: Path | str = _DEFAULT_EMBEDDING_CACHE_PATH) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _delete_stale_chunks(current_chunk_ids: List[str], document_id_prefixes: List[str], dsn: Optional[str], table_name: str) -> None:
    """Remove pgvector rows that belong to a document we just re-indexed but
    whose chunk_id is no longer among its current chunks (e.g. the document
    got shorter, or re-chunking produced fewer/merged chunks). Upserting
    alone (ON CONFLICT DO UPDATE) never removes rows, so without this,
    stale chunks from an old version of a document would linger in
    pgvector and keep showing up in retrieval forever.

    Scoped to `document_id_prefixes` — the document_ids actually present in
    this run — so it never touches chunks belonging to documents that
    weren't part of this indexing pass.
    """
    if not document_id_prefixes:
        return
    conn = get_pgvector_connection(dsn)
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            # Table may not exist yet on a fresh DB — nothing stale to delete.
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
                (table_name,),
            )
            if not cur.fetchone()[0]:
                return
            for prefix in document_id_prefixes:
                like_pattern = f"{prefix}_c%"
                if current_chunk_ids:
                    cur.execute(
                        f"DELETE FROM {table_name} WHERE chunk_id LIKE %s AND chunk_id != ALL(%s)",
                        (like_pattern, current_chunk_ids),
                    )
                else:
                    cur.execute(f"DELETE FROM {table_name} WHERE chunk_id LIKE %s", (like_pattern,))
            conn.commit()
    finally:
        conn.close()


def _embed_chunk_records(
    chunk_records: List[Dict[str, Any]],
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
    use_cache: bool = True,
) -> tuple[LocalVectorIndex, List[Dict[str, Any]], int, int, int]:
    """Shared core: embed a list of already-parsed chunk dicts (from
    chunks.jsonl) into a LocalVectorIndex + a list of pgvector-ready record
    dicts. Used by both index_chunks() and index_chunks_by_department().

    Incremental behavior: if `cache` is given and `use_cache` is True, a
    chunk whose chunk_id is already in the cache AND whose content_hash
    matches is reused as-is (no embedding call) — this is what makes
    re-indexing after a small edit cheap instead of re-embedding the whole
    corpus every time. `cache` is mutated in place with fresh entries so
    the caller can persist it after all departments/batches are done.

    Returns (index, records, llm_count, fallback_count, reused_count).
    """
    index = LocalVectorIndex(dim=768)
    records: List[Dict[str, Any]] = []
    llm_count = 0
    fallback_count = 0
    reused_count = 0
    if cache is None:
        cache = {}

    for record in chunk_records:
        metadata = {
            "department": record.get("department", "General"),
            "doc_type": record.get("doc_type", "Document"),
            "sensitivity": record.get("sensitivity", "Public"),
            "source_file": record.get("source_file", "unknown"),
            "section": record.get("section", "Document"),
        }
        content = record.get("content", "")
        chunk_id = record["chunk_id"]
        content_hash = record.get("content_hash") or _content_hash(content)

        cached_entry = cache.get(chunk_id) if use_cache else None
        if cached_entry is not None and cached_entry.get("content_hash") == content_hash:
            vector = cached_entry["embedding"]
            source = cached_entry.get("embedding_source", "llm")
            reused_count += 1
        else:
            vector, source = build_embedding(content, dim=768)
            cache[chunk_id] = {
                "content_hash": content_hash,
                "embedding": vector,
                "embedding_source": source,
            }
            if source == "llm":
                llm_count += 1
            else:
                fallback_count += 1

        index._ids.append(chunk_id)
        index._vectors.append(vector)
        index._payloads.append({"chunk_id": chunk_id, "content": content, "metadata": metadata, "embedding_source": source})
        records.append({
            "chunk_id": chunk_id,
            "embedding": vector,
            "content": content,
            "metadata": metadata,
            "department": metadata.get("department", "General"),
            "sensitivity": metadata.get("sensitivity", "Public"),
            "created_at": record.get("created_at", ""),
        })
    return index, records, llm_count, fallback_count, reused_count


def _report_embedding_stats(llm_count: int, fallback_count: int, label: str = "", reused_count: int = 0) -> None:
    prefix = f"[{label}] " if label else ""
    total = llm_count + fallback_count + reused_count
    if total == 0:
        print(f"{prefix}ℹ️  No chunks to embed.")
        return
    if reused_count:
        print(f"{prefix}♻️  {reused_count}/{total} chunks unchanged since last index — reused cached embeddings, "
              f"no re-embedding call made.")
    newly_embedded = llm_count + fallback_count
    if newly_embedded == 0:
        return
    if fallback_count > 0:
        print(f"{prefix}⚠️  {fallback_count}/{newly_embedded} newly-embedded chunks used the hash fallback "
              f"(no real embedding model reachable) — retrieval quality on these will be poor. "
              f"Check Ollama/litellm connectivity before trusting this index.")
    else:
        print(f"{prefix}✅ {llm_count} new/changed chunk(s) embedded via a real model.")


def _write_local_index_file(index: LocalVectorIndex, llm_count: int, fallback_count: int, index_file: Path) -> Path:
    index_file.write_text(json.dumps({
        "dim": index.dim,
        "ids": index._ids,
        "vectors": index._vectors,
        "payloads": index._payloads,
        "embedding_stats": {"llm": llm_count, "hash_fallback": fallback_count},
    }, indent=2), encoding="utf-8")
    return index_file


def _read_chunk_records(chunks_path: Path) -> List[Dict[str, Any]]:
    records = []
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def index_chunks(
    chunks_file: Path | str,
    output_dir: Path | str,
    pgvector_dsn: Optional[str] = None,
    table_name: str = "rag_chunks",
    use_cache: bool = True,
    cache_path: Path | str = _DEFAULT_EMBEDDING_CACHE_PATH,
) -> Path:
    """Build ONE combined local index + ONE pgvector table from all chunks,
    regardless of department. For real per-department storage (the "Two
    Department Knowledge Bases" spec deliverable), use
    index_chunks_by_department() instead.

    Incremental by default: chunks whose chunk_id + content_hash already
    exist in the embedding cache are reused instead of re-embedded, and
    stale rows for documents present in this run are deleted from pgvector.
    Pass use_cache=False to force a full re-embed of everything (e.g. after
    switching embedding models, where old cached vectors are no longer
    comparable to new ones)."""
    chunks_path = Path(chunks_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunk_records = _read_chunk_records(chunks_path)
    cache = _load_embedding_cache(cache_path) if use_cache else {}
    index, records, llm_count, fallback_count, reused_count = _embed_chunk_records(
        chunk_records, cache=cache, use_cache=use_cache
    )

    document_ids = sorted({r.get("document_id") for r in chunk_records if r.get("document_id")})
    current_chunk_ids = [r["chunk_id"] for r in records]
    _delete_stale_chunks(current_chunk_ids, document_ids, pgvector_dsn, table_name)

    _store_pgvector_records(records, pgvector_dsn, table_name=table_name)
    _report_embedding_stats(llm_count, fallback_count, reused_count=reused_count)
    if use_cache:
        _save_embedding_cache(cache, cache_path)

    index_file = output_path / "local_index.json"
    return _write_local_index_file(index, llm_count, fallback_count, index_file)


def index_chunks_by_department(
    chunks_file: Path | str,
    output_dir: Path | str,
    pgvector_dsn: Optional[str] = None,
    use_cache: bool = True,
    cache_path: Path | str = _DEFAULT_EMBEDDING_CACHE_PATH,
) -> Dict[str, Path]:
    """Build one local index file AND one pgvector table PER DEPARTMENT —
    real separate storage/collections, not a shared table filtered by a
    WHERE clause. This is what "Two Department Knowledge Bases" in the spec
    actually means; see access_control.department_table_name for how a
    department name maps to a table name (and why that mapping is
    deliberately strict).

    Each department gets its own local index file, named
    "local_index_<department_lower>.json" — NOT a shared "local_index.json"
    that every department's call would silently overwrite one after
    another. That overwrite is exactly the kind of bug this project already
    caught once before (see index_cli's single fixed output filename), so
    this function is built from the start to avoid it, rather than patched
    after the fact.

    Returns a dict of {department: index_file_path} for every department
    that had at least one chunk, so callers (e.g. a manifest writer) know
    exactly which department indexes were actually produced.

    Incremental by default, same as index_chunks() — see that function's
    docstring for how the embedding cache and stale-row cleanup work.
    """
    from src.retrieval.access_control import department_table_name, CANONICAL_DEPARTMENTS

    chunks_path = Path(chunks_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunk_records = _read_chunk_records(chunks_path)
    cache = _load_embedding_cache(cache_path) if use_cache else {}

    # Group chunks by department BEFORE embedding, so each department's
    # local index and pgvector table only ever contains that department's
    # own chunks — no cross-department leakage at the storage layer itself,
    # on top of the existing query-time department filter.
    by_department: Dict[str, List[Dict[str, Any]]] = {}
    unknown_departments: set = set()
    for record in chunk_records:
        department = record.get("department", "General")
        if department not in CANONICAL_DEPARTMENTS:
            # Don't silently drop or silently dump into "General" — that
            # would hide a real classification/validation gap. Collect and
            # report it clearly instead.
            unknown_departments.add(department)
            continue
        by_department.setdefault(department, []).append(record)

    if unknown_departments:
        print(f"⚠️  {len(unknown_departments)} unrecognized department value(s) skipped entirely "
              f"(not written to any department index): {sorted(unknown_departments)}. "
              f"These chunks should have been caught by rule-based validation before reaching "
              f"indexing — check src/classification/rule_validator.py's needs_review flag.")

    index_files: Dict[str, Path] = {}
    for department, department_records in sorted(by_department.items()):
        index, records, llm_count, fallback_count, reused_count = _embed_chunk_records(
            department_records, cache=cache, use_cache=use_cache
        )

        table_name = department_table_name(department)
        document_ids = sorted({r.get("document_id") for r in department_records if r.get("document_id")})
        current_chunk_ids = [r["chunk_id"] for r in records]
        _delete_stale_chunks(current_chunk_ids, document_ids, pgvector_dsn, table_name)

        _store_pgvector_records(records, pgvector_dsn, table_name=table_name)
        _report_embedding_stats(llm_count, fallback_count, label=department, reused_count=reused_count)

        index_file = output_path / f"local_index_{department.lower()}.json"
        index_files[department] = _write_local_index_file(index, llm_count, fallback_count, index_file)
        print(f"  [{department}] {len(department_records)} chunks → table '{table_name}', "
              f"local index '{index_file.name}'")

    # A small manifest so retrieval/dashboard code can discover which
    # department indexes exist without re-scanning chunks.jsonl itself.
    manifest_file = output_path / "department_index_manifest.json"
    manifest_file.write_text(json.dumps({
        "departments": {
            department: {
                "table_name": department_table_name(department),
                "local_index_file": str(path.name),
                "chunk_count": len(by_department[department]),
            }
            for department, path in index_files.items()
        },
        "unknown_departments_skipped": sorted(unknown_departments),
    }, indent=2), encoding="utf-8")

    if use_cache:
        _save_embedding_cache(cache, cache_path)

    return index_files
