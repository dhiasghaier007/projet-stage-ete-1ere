"""
Lexical (BM25) search and Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Hybrid retrieval combines two independent rankings of the same corpus:
  - semantic: embedding cosine similarity (catches meaning/paraphrase)
  - lexical:  BM25 over raw tokens (catches exact terms, IDs, names, codes
              that embeddings tend to blur together)

RRF fuses the two rankings without needing to normalize or calibrate their
raw scores against each other (cosine similarity and BM25 scores are not on
comparable scales) — it only uses each result's *rank position* in each list.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.access_control import DEFAULT_CLEARANCE, filter_chunks_by_clearance

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency may be absent
    _BM25_AVAILABLE = False


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def build_lexical_index(payloads: List[Dict[str, Any]]) -> Optional[Tuple[Any, List[str]]]:
    """Build a BM25 index over chunk payloads. Returns (bm25, ordered_chunk_ids) or
    None if rank_bm25 isn't installed — callers must treat None as "lexical search
    unavailable" and fall back to semantic-only, loudly, not silently."""
    if not _BM25_AVAILABLE:
        return None
    if not payloads:
        return None
    corpus = [_tokenize(p["content"]) for p in payloads]
    ids = [p["chunk_id"] for p in payloads]
    return BM25Okapi(corpus), ids


def lexical_search(bm25_index: Tuple[Any, List[str]], query: str, top_k: int) -> List[str]:
    """Return chunk_ids ranked by BM25 score, best first."""
    bm25, ids = bm25_index
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(ids, scores), key=lambda item: item[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked[:top_k] if score > 0]


def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Standard RRF: score(id) = sum over lists of 1 / (k + rank), rank is 1-indexed.

    k=60 is the commonly used default from the original RRF paper — it dampens
    the influence of any single list's top rank so neither retriever dominates
    just by having a very confident #1 result.
    """
    scores: Dict[str, float] = {}
    for ranked_list in ranked_lists:
        for rank, chunk_id in enumerate(ranked_list, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


def hybrid_search(
    index,  # LocalVectorIndex
    query: str,
    top_k: int = 5,
    candidate_k: int = 30,
    rrf_k: int = 60,
    clearance: str = DEFAULT_CLEARANCE,
) -> Tuple[List[Dict[str, Any]], str]:
    """Fuse semantic + lexical rankings via RRF. Returns (results, retrieval_mode)
    where retrieval_mode is "hybrid_rrf" or "semantic_only" (if BM25 unavailable) —
    surfaced explicitly rather than silently degrading to one retriever.

    `clearance` enforces sensitivity-based access control: chunks the caller
    isn't cleared for are removed BEFORE ranking/fusion, not filtered out of
    an already-ranked result afterward. This matters because even just
    knowing a Restricted chunk ranked highly for a query is itself a signal
    ("something confidential exists about X") — so the disallowed chunks
    never enter the candidate pool that produces rrf_score or rank position
    in the first place.
    """
    allowed_payloads = filter_chunks_by_clearance(index._payloads, clearance)

    semantic_candidates_raw = index.search(query, top_k=candidate_k)
    semantic_candidates = filter_chunks_by_clearance(semantic_candidates_raw, clearance)
    semantic_rank_list = [item["chunk_id"] for item in semantic_candidates]
    by_id = {item["chunk_id"]: item for item in semantic_candidates}

    bm25_index = build_lexical_index(allowed_payloads)
    if bm25_index is None:
        print("⚠️  rank_bm25 not installed or corpus empty — falling back to semantic-only retrieval "
              "(lexical/exact-term matching is disabled).")
        return semantic_candidates[:top_k], "semantic_only"

    lexical_rank_list = lexical_search(bm25_index, query, top_k=candidate_k)
    for chunk_id in lexical_rank_list:
        if chunk_id not in by_id:
            payload = next((p for p in allowed_payloads if p["chunk_id"] == chunk_id), None)
            if payload:
                by_id[chunk_id] = {"chunk_id": chunk_id, "score": 0.0, "content": payload["content"], "metadata": payload["metadata"]}

    fused_scores = reciprocal_rank_fusion([semantic_rank_list, lexical_rank_list], k=rrf_k)
    ranked_ids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

    results = []
    for chunk_id in ranked_ids:
        item = dict(by_id[chunk_id])
        item["rrf_score"] = round(fused_scores[chunk_id], 6)
        item["in_semantic_results"] = chunk_id in semantic_rank_list
        item["in_lexical_results"] = chunk_id in lexical_rank_list
        results.append(item)
    return results, "hybrid_rrf"
