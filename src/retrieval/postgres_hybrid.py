"""
Postgres-backed hybrid retrieval: pgvector for semantic search, native
full-text search (tsvector/GIN) for lexical search, fused with the same
reciprocal_rank_fusion() used by the local (in-memory) path — so results are
consistent regardless of which backend actually served the query.
"""
from typing import Any, Dict, List, Optional, Tuple

from src.indexing.index_vectors import build_embedding, get_pgvector_connection
from src.retrieval.hybrid_search import reciprocal_rank_fusion, _tokenize
from src.retrieval.access_control import DEFAULT_CLEARANCE, allowed_sensitivity_labels, filter_chunks_by_clearance


def _to_or_tsquery(text: str) -> str:
    """Build a tsquery string that matches ANY of the question's terms, not all
    of them. plainto_tsquery ANDs every word together — with the 'simple' text
    search config (no stopword removal, needed for FR/AR support), a question
    like "What is X about?" would require "what", "is", and "about" to all
    literally appear in the same chunk, which almost never happens. OR is the
    correct semantics for "which chunks are relevant to any of these terms."
    """
    tokens = _tokenize(text)
    return " | ".join(tokens) if tokens else ""


def _semantic_rank_pg(conn, query_vector: List[float], top_k: int, table_name: str, allowed_labels: List[str]) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT chunk_id FROM {table_name} WHERE sensitivity = ANY(%s) ORDER BY embedding <=> %s::vector LIMIT %s",
            (allowed_labels, query_vector, top_k),
        )
        return [row[0] for row in cur.fetchall()]


def _lexical_rank_pg(conn, query_text: str, top_k: int, table_name: str, allowed_labels: List[str]) -> List[str]:
    or_query = _to_or_tsquery(query_text)
    if not or_query:
        return []
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT chunk_id FROM {table_name}
            WHERE content_tsv @@ to_tsquery('simple', %s) AND sensitivity = ANY(%s)
            ORDER BY ts_rank_cd(content_tsv, to_tsquery('simple', %s)) DESC
            LIMIT %s
            """,
            (or_query, allowed_labels, or_query, top_k),
        )
        return [row[0] for row in cur.fetchall()]


def _fetch_payloads_pg(conn, chunk_ids: List[str], table_name: str) -> Dict[str, Dict[str, Any]]:
    if not chunk_ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT chunk_id, content, metadata FROM {table_name} WHERE chunk_id = ANY(%s)",
            (chunk_ids,),
        )
        return {row[0]: {"content": row[1], "metadata": row[2]} for row in cur.fetchall()}


def hybrid_search_pg(
    query: str,
    top_k: int = 5,
    candidate_k: int = 30,
    rrf_k: int = 60,
    dsn: Optional[str] = None,
    table_name: str = "rag_chunks",
    clearance: str = DEFAULT_CLEARANCE,
) -> Optional[Tuple[List[Dict[str, Any]], str]]:
    """Run hybrid search against Postgres. Returns None (not an empty result!)
    if Postgres isn't reachable at all, so callers can distinguish "DB down,
    fall back to local index" from "DB up, genuinely no matches".

    `clearance` is enforced as a SQL WHERE filter on both the semantic and
    lexical queries, so disallowed rows never leave the database at all —
    the strongest form of enforcement available here (unlike the local JSON
    index path, which has to filter in Python after search since it has no
    query-time filtering of its own). A second, redundant filter is applied
    to the final results below as defense-in-depth in case the two policies
    (SQL allowed_labels vs. access_control.is_allowed) ever drift.
    """
    conn = get_pgvector_connection(dsn)
    if conn is None:
        return None

    allowed_labels = allowed_sensitivity_labels(clearance)

    try:
        query_vector, embed_source = build_embedding(query, dim=768)
        if embed_source == "hash_fallback":
            print("⚠️  Query embedding used the hash fallback even with Postgres reachable — "
                  "semantic ranking will be unreliable.")

        semantic_ids = _semantic_rank_pg(conn, query_vector, candidate_k, table_name, allowed_labels)
        lexical_ids = _lexical_rank_pg(conn, query, candidate_k, table_name, allowed_labels)

        fused_scores = reciprocal_rank_fusion([semantic_ids, lexical_ids], k=rrf_k)
        ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

        payloads = _fetch_payloads_pg(conn, ranked_ids, table_name)
        results = []
        for chunk_id in ranked_ids:
            payload = payloads.get(chunk_id, {"content": "", "metadata": {}})
            results.append({
                "chunk_id": chunk_id,
                "rrf_score": round(fused_scores[chunk_id], 6),
                "content": payload["content"],
                "metadata": payload["metadata"],
                "in_semantic_results": chunk_id in semantic_ids,
                "in_lexical_results": chunk_id in lexical_ids,
            })
        results = filter_chunks_by_clearance(results, clearance)
        return results, "hybrid_rrf_postgres"
    finally:
        conn.close()
