"""
Tests for qa_cli.py's retrieval-visibility output (the --verbose / "verbose
on" feature). These are pure formatting tests against a constructed fake
`answer_question()` result — no live LLM or index needed, since the thing
being tested is "does the CLI print the right information", not "does
retrieval work" (that's covered in test_rag_quality.py).
"""
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qa.qa_cli import _print_retrieval_details


def _fake_result(**overrides) -> dict:
    base = {
        "question": "who approves it?",
        "retrieval_query": "who approves the remote work policy?",
        "query_rewrite_status": "rewritten",
        "retrieval_mode": "hybrid_rrf",
        "answer": "The direct manager and HR Director approve remote work requests.",
        "retrieved_chunks": [
            {
                "chunk_id": "chunk_00000",
                "rrf_score": 0.0328,
                "content": "REMOTE WORK POLICY — Effective January 2026. Requests must be approved by the direct manager and HR Director.",
                "metadata": {"source_file": "doc_001_hr_policy_english.md", "sensitivity": "Internal"},
            },
        ],
    }
    base.update(overrides)
    return base


class RetrievalVisibilityTests(unittest.TestCase):

    def test_shows_rewritten_query_when_status_is_rewritten(self):
        result = _fake_result()
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_retrieval_details(result)
        output = buf.getvalue()
        self.assertIn("who approves the remote work policy?", output)
        self.assertIn("who approves it?", output)

    def test_does_not_mention_rewriting_when_status_is_no_history(self):
        # First turn of a session — nothing was rewritten, and the output
        # shouldn't imply otherwise or clutter the display with a no-op.
        result = _fake_result(query_rewrite_status="no_history", retrieval_query="who approves it?")
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_retrieval_details(result)
        output = buf.getvalue()
        self.assertNotIn("Rewritten query", output)

    def test_does_not_mention_rewriting_when_question_already_standalone(self):
        result = _fake_result(query_rewrite_status="unnecessary", retrieval_query="who approves it?")
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_retrieval_details(result)
        output = buf.getvalue()
        self.assertNotIn("Rewritten query", output)

    def test_warns_when_rewrite_failed_instead_of_hiding_it(self):
        # If rewriting genuinely failed (not just skipped), the person
        # should be told retrieval used the raw, possibly-ambiguous
        # question — silently proceeding would hide a real degradation.
        for failure_status in ("unavailable", "unparseable", "error"):
            with self.subTest(status=failure_status):
                result = _fake_result(query_rewrite_status=failure_status, retrieval_query="who approves it?")
                buf = io.StringIO()
                with redirect_stdout(buf):
                    _print_retrieval_details(result)
                output = buf.getvalue()
                self.assertIn(failure_status, output)
                self.assertIn("did not run", output)

    def test_shows_retrieval_mode(self):
        result = _fake_result(retrieval_mode="hybrid_rrf_postgres")
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_retrieval_details(result)
        self.assertIn("hybrid_rrf_postgres", buf.getvalue())

    def test_shows_chunk_score_sensitivity_and_source(self):
        result = _fake_result()
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_retrieval_details(result)
        output = buf.getvalue()
        self.assertIn("0.0328", output)
        self.assertIn("Internal", output)
        self.assertIn("doc_001_hr_policy_english.md", output)

    def test_long_chunk_content_is_truncated_for_display(self):
        long_content = "A" * 500
        result = _fake_result(retrieved_chunks=[{
            "chunk_id": "c1", "rrf_score": 0.01, "content": long_content,
            "metadata": {"source_file": "big.md", "sensitivity": "Public"},
        }])
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_retrieval_details(result)
        output = buf.getvalue()
        # Full 500-char content must not appear verbatim — this is a
        # terminal preview, not a dump of the whole chunk.
        self.assertNotIn("A" * 500, output)
        self.assertIn("...", output)

    def test_handles_empty_retrieval_gracefully(self):
        result = _fake_result(retrieved_chunks=[], query_rewrite_status="no_history", retrieval_query="what color is the sky?")
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_retrieval_details(result)
        output = buf.getvalue()
        self.assertIn("No chunks retrieved", output)

    def test_missing_metadata_fields_do_not_crash(self):
        # A chunk with no source_file/sensitivity (shouldn't normally
        # happen post-classification, but the display must not crash if it does).
        result = _fake_result(retrieved_chunks=[{
            "chunk_id": "c1", "rrf_score": 0.01, "content": "some content",
            "metadata": {},
        }])
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_retrieval_details(result)  # should not raise
        output = buf.getvalue()
        self.assertIn("unlabeled", output)
        self.assertIn("unknown source", output)


if __name__ == "__main__":
    unittest.main()
