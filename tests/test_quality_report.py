import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.qa.quality_report import (
    _chunk_health_score,
    _department_coverage_score,
    _embedding_coverage_score,
    _sensitivity_label_validity_score,
    compute_corpus_quality_score,
    detect_drift,
    save_report,
)


def _payload(content, department="HR", sensitivity="Public"):
    return {"content": content, "metadata": {"department": department, "sensitivity": sensitivity}}


class MetricUnitTests(unittest.TestCase):
    """Each sub-metric is a small, deterministic calculation over a fake
    index dict — these don't need a real index file or any live model, so
    they run everywhere, unlike the retrieval-regression metric which needs
    a real index."""

    def test_embedding_coverage_reports_real_vs_fallback_split(self):
        index = {"embedding_stats": {"llm": 3, "hash_fallback": 1}}
        result = _embedding_coverage_score(index)
        self.assertEqual(result["score"], 75.0)

    def test_embedding_coverage_handles_empty_index(self):
        index = {"embedding_stats": {"llm": 0, "hash_fallback": 0}}
        result = _embedding_coverage_score(index)
        self.assertIsNone(result["score"])

    def test_sensitivity_validity_flags_unrecognized_labels(self):
        index = {"payloads": [
            _payload("a", sensitivity="Public"),
            _payload("b", sensitivity="TOP_SECRET_MADE_UP"),
        ]}
        result = _sensitivity_label_validity_score(index)
        self.assertEqual(result["score"], 50.0)

    def test_department_coverage_excludes_default_general(self):
        index = {"payloads": [
            _payload("a", department="HR"),
            _payload("b", department="General"),
            _payload("c", department=""),
        ]}
        result = _department_coverage_score(index)
        self.assertAlmostEqual(result["score"], 100.0 / 3, places=1)

    def test_chunk_health_flags_tiny_and_duplicate_chunks(self):
        index = {"payloads": [
            _payload("this is a real chunk with enough words to count"),
            _payload("too short"),
            _payload("this is a real chunk with enough words to count"),  # exact duplicate
        ]}
        result = _chunk_health_score(index)
        # 1 healthy out of 3 (one too-short, one duplicate)
        self.assertAlmostEqual(result["score"], 100.0 / 3, places=1)

    def test_overall_score_excludes_uncomputable_metrics_from_average(self):
        # An empty index means every content-based metric returns score=None
        # (nothing to compute) — overall_score must not treat those as 0s,
        # since that would make an empty index look "bad" rather than "N/A".
        with tempfile.TemporaryDirectory() as tmp:
            empty_index_path = Path(tmp) / "empty_index.json"
            empty_index_path.write_text(json.dumps({
                "dim": 8, "ids": [], "vectors": [], "payloads": [],
                "embedding_stats": {"llm": 0, "hash_fallback": 0},
            }))
            report = compute_corpus_quality_score(empty_index_path)
            # Every metric should be None (nothing to score), so overall
            # should also end up None, not a fabricated 0.
            for metric in report["metrics"].values():
                self.assertIsNone(metric["score"])
            self.assertIsNone(report["overall_score"])


class DriftDetectionTests(unittest.TestCase):
    """Verifies detect_drift's three real branches — no_baseline, stable,
    and drift — actually distinguish between these cases rather than
    always returning the same status regardless of input."""

    def test_no_baseline_on_first_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty_reports_dir = Path(tmp)
            current = {"metrics": {"foo": {"score": 90.0}}}
            result = detect_drift(current, empty_reports_dir)
            self.assertEqual(result["status"], "no_baseline")

    def test_stable_when_no_metric_drops_past_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp)
            previous = {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "metrics": {"embedding_coverage": {"score": 90.0}},
            }
            save_report(previous, reports_dir)
            current = {"metrics": {"embedding_coverage": {"score": 88.0}}}  # small drop, within threshold
            result = detect_drift(current, reports_dir)
            self.assertEqual(result["status"], "stable")

    def test_drift_flagged_when_metric_drops_past_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp)
            previous = {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "metrics": {"embedding_coverage": {"score": 90.0}},
            }
            save_report(previous, reports_dir)
            current = {"metrics": {"embedding_coverage": {"score": 40.0}}}  # big drop, past threshold
            result = detect_drift(current, reports_dir)
            self.assertEqual(result["status"], "drift")
            self.assertEqual(len(result["regressions"]), 1)
            self.assertEqual(result["regressions"][0]["metric"], "embedding_coverage")

    def test_drift_ignores_metrics_missing_from_either_side(self):
        # A metric that's new (wasn't in the previous report) or that
        # returned score=None on either side must not crash or falsely
        # trigger drift — it should just be skipped from comparison.
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp)
            previous = {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "metrics": {"embedding_coverage": {"score": 90.0}},
            }
            save_report(previous, reports_dir)
            current = {"metrics": {
                "embedding_coverage": {"score": 90.0},
                "brand_new_metric": {"score": 10.0},
                "uncomputable_metric": {"score": None},
            }}
            result = detect_drift(current, reports_dir)
            self.assertEqual(result["status"], "stable")
            self.assertNotIn("brand_new_metric", result["metrics_checked"])


if __name__ == "__main__":
    unittest.main()
