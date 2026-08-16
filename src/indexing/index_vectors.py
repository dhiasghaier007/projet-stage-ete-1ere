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


def index_chunks(chunks_file: Path | str, output_dir: Path | str, pgvector_dsn: Optional[str] = None, table_name: str = "rag_chunks") -> Path:
    chunks_path = Path(chunks_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    index = LocalVectorIndex(dim=768)
    records: List[Dict[str, Any]] = []
    llm_count = 0
    fallback_count = 0
    with chunks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            metadata = {
                "department": record.get("department", "General"),
                "doc_type": record.get("doc_type", "Document"),
                "sensitivity": record.get("sensitivity", "Public"),
                "source_file": record.get("source_file", "unknown"),
                "section": record.get("section", "Document"),
            }
            content = record.get("content", "")
            vector, source = build_embedding(content, dim=768)
            if source == "llm":
                llm_count += 1
            else:
                fallback_count += 1
            index.add_chunk(record["chunk_id"], content, metadata)
            records.append({
                "chunk_id": record["chunk_id"],
                "embedding": vector,
                "content": content,
                "metadata": metadata,
                "department": metadata.get("department", "General"),
                "sensitivity": metadata.get("sensitivity", "Public"),
                "created_at": record.get("created_at", ""),
            })

    _store_pgvector_records(records, pgvector_dsn, table_name=table_name)

    if fallback_count > 0:
        print(f"⚠️  {fallback_count}/{llm_count + fallback_count} chunks used the hash fallback "
              f"(no real embedding model reachable) — retrieval quality on these will be poor. "
              f"Check Ollama/litellm connectivity before trusting this index.")
    else:
        print(f"✅ All {llm_count} chunks embedded via a real model.")

    index_file = output_path / "local_index.json"
    index_file.write_text(json.dumps({
        "dim": index.dim,
        "ids": index._ids,
        "vectors": index._vectors,
        "payloads": index._payloads,
        "embedding_stats": {"llm": llm_count, "hash_fallback": fallback_count},
    }, indent=2), encoding="utf-8")
    return index_file
