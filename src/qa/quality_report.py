"""
Corpus Quality Score + Drift Detection — Stage 5 QA additions.

Two things this module does, deliberately kept separate from
ragas_eval.py's per-answer faithfulness judging (that's about whether one
generated answer is grounded in its context; this is about whether the
INDEX ITSELF is healthy — is retrieval likely to work well right now,
regardless of any single question):

1. compute_corpus_quality_score(): a single 0-100 score built from cheap,
   deterministic checks on the index (no LLM call needed, so this can run
   even when Ollama/litellm are unreachable) plus a small labeled retrieval
   regression set (reusing the same (question -> expected_doc) pairs the
   test suite uses, so "quality" here means the same thing "passing tests"
   means, not a separate ad-hoc definition).

2. detect_drift(): compares a freshly computed report against the most
   recently saved one and flags any metric that dropped by more than a
   threshold. This is corpus/index drift (did re-ingesting or re-chunking
   quietly make retrieval worse?), not model drift — a different, narrower
   claim than "detect if the LLM's behavior changed," which nothing in
   this codebase attempts to measure and this module does not claim to
   either.

Design choices mirroring the rest of the pipeline:
- No silent fallback. A metric that can't be computed (e.g. no regression
  set matched anything) is reported as null/skipped, never averaged in as
  if it were a real 0.
- Every report is saved with a timestamp so this builds a real trend over
  time, not just a snapshot.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.retrieval.access_control import SENSITIVITY_LEVELS

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "indexing" / "local_index.json"
_REPORTS_DIR = REPO_ROOT / "data" / "outputs" / "quality_reports"

# Same labeled set the offline regression test uses (tests/test_rag_quality.py
# _REGRESSION_SET) — kept in sync manually since the test file isn't meant to
# be imported by production code. If you add cases to one, add them to the
# other.
_REGRESSION_SET = [
    ("What is the remote work policy about?", "doc_001_hr_policy_english"),
    ("What was the system uptime last week?", "doc_004_it_report_english"),
    ("When will the office close for system maintenance?", "doc_005_general_email_french_arabic"),
]

# A metric dropping by more than this many points (on its own 0-100 scale)
# between consecutive reports is flagged as drift. Deliberately a single
# flat threshold rather than per-metric tuning — simple enough to reason
# about, and easy to tighten later if it proves too noisy or too quiet.
_DRIFT_THRESHOLD_POINTS = 5.0


def _load_index(index_path: Path) -> Optional[Dict[str, Any]]:
    if not index_path.exists():
        return None
    return json.loads(index_path.read_text(encoding="utf-8"))


def _embedding_coverage_score(index: Dict[str, Any]) -> Dict[str, Any]:
    """% of chunks embedded via a real model rather than the hash fallback.
    The hash fallback produces vectors with no real semantic meaning, so a
    corpus leaning on it heavily will retrieve poorly regardless of how
    good the chunking/classification upstream was."""
    stats = index.get("embedding_stats", {})
    llm = stats.get("llm", 0)
    fallback = stats.get("hash_fallback", 0)
    total = llm + fallback
    if total == 0:
        return {"score": None, "detail": "no chunks in index"}
    score = 100.0 * llm / total
    return {"score": round(score, 1), "detail": f"{llm}/{total} chunks used a real embedding model"}


def _sensitivity_label_validity_score(index: Dict[str, Any]) -> Dict[str, Any]:
    """% of chunks carrying a recognized sensitivity label. An unrecognized
    or missing label is treated as maximally restricted at query time (see
    access_control.py), which is the safe failure mode — but it also means
    that chunk becomes invisible to anyone below Restricted clearance, so a
    high rate of this happening silently shrinks the searchable corpus."""
    payloads = index.get("payloads", [])
    if not payloads:
        return {"score": None, "detail": "no chunks in index"}
    valid = sum(
        1 for p in payloads
        if p.get("metadata", {}).get("sensitivity") in SENSITIVITY_LEVELS
    )
    score = 100.0 * valid / len(payloads)
    return {"score": round(score, 1), "detail": f"{valid}/{len(payloads)} chunks have a recognized sensitivity label"}


def _department_coverage_score(index: Dict[str, Any]) -> Dict[str, Any]:
    """% of chunks tagged with a real department (not the 'General' default,
    which usually means classification didn't confidently assign one)."""
    payloads = index.get("payloads", [])
    if not payloads:
        return {"score": None, "detail": "no chunks in index"}
    tagged = sum(
        1 for p in payloads
        if p.get("metadata", {}).get("department") not in (None, "", "General")
    )
    score = 100.0 * tagged / len(payloads)
    departments = sorted({p.get("metadata", {}).get("department", "General") for p in payloads})
    return {
        "score": round(score, 1),
        "detail": f"{tagged}/{len(payloads)} chunks have a specific (non-default) department",
        "departments_seen": departments,
    }


def _chunk_health_score(index: Dict[str, Any]) -> Dict[str, Any]:
    """% of chunks that aren't suspiciously empty/tiny (a sign of a chunking
    or upstream extraction bug, not a real gap in the source document) and
    aren't exact-duplicate content that slipped past chunking.py's dedup."""
    payloads = index.get("payloads", [])
    if not payloads:
        return {"score": None, "detail": "no chunks in index"}
    seen_content: set = set()
    healthy = 0
    tiny = 0
    duplicate = 0
    for p in payloads:
        content = (p.get("content") or "").strip()
        if len(content.split()) < 5:
            tiny += 1
            continue
        normalized = " ".join(content.split()).lower()
        if normalized in seen_content:
            duplicate += 1
            continue
        seen_content.add(normalized)
        healthy += 1
    total = len(payloads)
    score = 100.0 * healthy / total
    return {
        "score": round(score, 1),
        "detail": f"{healthy}/{total} chunks healthy ({tiny} too short, {duplicate} duplicate content)",
    }


def _retrieval_regression_score(index_path: Path, clearance: str = "Restricted") -> Dict[str, Any]:
    """Runs the same labeled (question -> expected source document) set the
    test suite uses, but through pure retrieval (no LLM/generation call), so
    this metric is available even when Ollama/litellm are unreachable —
    unlike answer-level faithfulness judging, which needs a live model."""
    from src.indexing.index_vectors import LocalVectorIndex, build_embedding
    from src.retrieval.hybrid_search import hybrid_search

    index_payload = _load_index(index_path)
    if index_payload is None:
        return {"score": None, "detail": f"no index found at {index_path}"}
    if not index_payload.get("payloads"):
        return {"score": None, "detail": "index exists but has no chunks — nothing to test retrieval against"}

    local_index = LocalVectorIndex(dim=index_payload["dim"])
    for chunk_id, vector, payload_item in zip(
        index_payload["ids"], index_payload["vectors"], index_payload["payloads"]
    ):
        local_index._ids.append(chunk_id)
        local_index._vectors.append(vector)
        local_index._payloads.append(payload_item)

    hits = 0
    misses: List[Dict[str, str]] = []
    for question, expected_doc in _REGRESSION_SET:
        results, _mode = hybrid_search(local_index, question, top_k=3, clearance=clearance)
        retrieved_files = [r["metadata"].get("source_file", "") for r in results]
        if any(expected_doc in f for f in retrieved_files):
            hits += 1
        else:
            misses.append({"question": question, "expected": expected_doc, "got": retrieved_files})

    total = len(_REGRESSION_SET)
    score = 100.0 * hits / total if total else None
    return {
        "score": round(score, 1) if score is not None else None,
        "detail": f"{hits}/{total} regression questions retrieved their expected document",
        "misses": misses,
    }


def compute_corpus_quality_score(index_path: Path | str = _DEFAULT_INDEX_PATH) -> Dict[str, Any]:
    """Compute the full corpus quality report. Returns a dict with each
    sub-metric plus an overall 0-100 score (the unweighted mean of whichever
    sub-metrics were actually computable — a metric that returned None
    because e.g. the index was empty is excluded from the average rather
    than silently counted as 0)."""
    index_path = Path(index_path)
    index = _load_index(index_path)

    metrics: Dict[str, Dict[str, Any]] = {}
    if index is None:
        metrics["index_found"] = {"score": 0.0, "detail": f"no index file at {index_path}"}
    else:
        metrics["embedding_coverage"] = _embedding_coverage_score(index)
        metrics["sensitivity_label_validity"] = _sensitivity_label_validity_score(index)
        metrics["department_coverage"] = _department_coverage_score(index)
        metrics["chunk_health"] = _chunk_health_score(index)
        metrics["retrieval_regression"] = _retrieval_regression_score(index_path)

    computable_scores = [m["score"] for m in metrics.values() if m.get("score") is not None]
    overall_score = round(sum(computable_scores) / len(computable_scores), 1) if computable_scores else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_path": str(index_path),
        "overall_score": overall_score,
        "metrics": metrics,
    }


def _load_previous_report(reports_dir: Path = _REPORTS_DIR) -> Optional[Dict[str, Any]]:
    if not reports_dir.exists():
        return None
    report_files = sorted(reports_dir.glob("quality_report_*.json"))
    if not report_files:
        return None
    return json.loads(report_files[-1].read_text(encoding="utf-8"))


def detect_drift(current_report: Dict[str, Any], reports_dir: Path = _REPORTS_DIR) -> Dict[str, Any]:
    """Compare `current_report` against the most recently saved report.
    Returns status "no_baseline" (nothing to compare against yet — first
    run), "stable" (no metric dropped past the threshold), or "drift"
    (one or more metrics regressed), always listing which metrics were
    checked so a "stable" result is visibly based on real comparisons, not
    just an absence of data."""
    previous = _load_previous_report(reports_dir)
    if previous is None:
        return {"status": "no_baseline", "detail": "no prior report to compare against — this is the first run"}

    regressions = []
    checked = []
    prev_metrics = previous.get("metrics", {})
    curr_metrics = current_report.get("metrics", {})
    for name, curr in curr_metrics.items():
        prev = prev_metrics.get(name)
        if prev is None or curr.get("score") is None or prev.get("score") is None:
            continue
        checked.append(name)
        delta = curr["score"] - prev["score"]
        if delta < -_DRIFT_THRESHOLD_POINTS:
            regressions.append({
                "metric": name,
                "previous_score": prev["score"],
                "current_score": curr["score"],
                "delta": round(delta, 1),
            })

    status = "drift" if regressions else "stable"
    return {
        "status": status,
        "compared_against": previous.get("generated_at"),
        "metrics_checked": checked,
        "regressions": regressions,
    }


def save_report(report: Dict[str, Any], reports_dir: Path = _REPORTS_DIR) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = reports_dir / f"quality_report_{timestamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return out_path


def run(index_path: Path | str = _DEFAULT_INDEX_PATH, reports_dir: Path = _REPORTS_DIR) -> Dict[str, Any]:
    """Full flow: compute the score, check drift against history, save,
    and return everything (used by both the CLI below and dashboard.py)."""
    report = compute_corpus_quality_score(index_path)
    drift = detect_drift(report, reports_dir)
    report["drift"] = drift
    saved_path = save_report(report, reports_dir)
    report["_saved_path"] = str(saved_path)
    return report


def _print_report(report: Dict[str, Any]) -> None:
    print("=" * 70)
    print("CORPUS QUALITY REPORT")
    print("=" * 70)
    overall = report.get("overall_score")
    print(f"Overall score: {overall if overall is not None else 'N/A'}/100")
    print(f"Generated at:  {report['generated_at']}")
    print()
    for name, metric in report.get("metrics", {}).items():
        score = metric.get("score")
        score_str = f"{score:5.1f}" if score is not None else "  N/A"
        print(f"  {name:28s}: {score_str}  — {metric.get('detail', '')}")

    print()
    drift = report.get("drift", {})
    status = drift.get("status")
    if status == "no_baseline":
        print("Drift check: no prior report to compare against (first run).")
    elif status == "stable":
        print(f"Drift check: STABLE — no metric regressed by more than "
              f"{_DRIFT_THRESHOLD_POINTS} points vs. {drift.get('compared_against')}.")
    elif status == "drift":
        print(f"Drift check: ⚠️  DRIFT DETECTED vs. {drift.get('compared_against')}:")
        for reg in drift.get("regressions", []):
            print(f"  {reg['metric']}: {reg['previous_score']} → {reg['current_score']} "
                  f"({reg['delta']:+.1f} points)")
    print("=" * 70)
    if "_saved_path" in report:
        print(f"Report saved to {report['_saved_path']}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute corpus quality score and check for drift.")
    parser.add_argument("--index", default=str(_DEFAULT_INDEX_PATH), help="Path to local_index.json")
    args = parser.parse_args()
    report = run(index_path=Path(args.index))
    _print_report(report)


if __name__ == "__main__":
    main()
