import json
import tempfile
import unittest
from pathlib import Path

from src.indexing.index_vectors import LocalVectorIndex, build_embedding, index_chunks
from src.qa.rag_pipeline import answer_question, evaluate_rag
from src.retrieval.hybrid_search import reciprocal_rank_fusion


class RetrievalTests(unittest.TestCase):
    def test_reciprocal_rank_fusion_rewards_items_ranked_high_in_both_lists(self):
        semantic_ranked = ["a", "b", "c"]
        lexical_ranked = ["b", "a", "c"]
        scores = reciprocal_rank_fusion([semantic_ranked, lexical_ranked], k=60)

        # 'a' is #1 semantic + #2 lexical, 'b' is #2 semantic + #1 lexical —
        # both appear once at rank 1 and once at rank 2, so they should tie
        self.assertAlmostEqual(scores["a"], scores["b"])
        # 'c' is #3 in both lists, so it must score strictly lower than both
        self.assertLess(scores["c"], scores["a"])

    def test_reciprocal_rank_fusion_rewards_agreement_over_a_single_strong_rank(self):
        # 'x' is #1 in one list but absent from the other; 'y' is #2 in both —
        # RRF should let consistent agreement beat a single very strong rank
        scores = reciprocal_rank_fusion([["x", "y"], ["z", "y"]], k=60)
        self.assertGreater(scores["y"], scores["x"])

    def test_build_embedding_returns_vector_with_expected_length(self):
        vector, source = build_embedding("remote work policy for employees")
        self.assertIsInstance(vector, list)
        self.assertGreater(len(vector), 10)
        self.assertIn(source, ("llm", "hash_fallback"))

    def test_local_vector_index_returns_relevant_chunk(self):
        index = LocalVectorIndex(dim=8)
        chunks = [
            {"chunk_id": "a", "content": "Remote work policy for employees", "metadata": {"department": "HR"}},
            {"chunk_id": "b", "content": "Invoice payment terms for finance", "metadata": {"department": "Finance"}},
        ]
        for chunk in chunks:
            index.add_chunk(chunk["chunk_id"], chunk["content"], chunk["metadata"])

        result = index.search("remote work policy", top_k=1)[0]
        self.assertEqual(result["chunk_id"], "a")

    def test_answer_question_and_evaluation_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            chunks_file = tmp_path / "chunks.jsonl"
            chunks_file.write_text(
                json.dumps({
                    "chunk_id": "chunk_000", "source_file": "doc.md", "department": "HR",
                    "doc_type": "Policy", "sensitivity": "Internal", "content": "Remote work policy allows flexible schedules.",
                    "section": "Overview", "chunk_size": 8, "created_at": "now"
                }) + "\n",
                encoding="utf-8",
            )
            output_dir = tmp_path / "out"
            # Deliberately point at an unreachable DSN, not the real one from
            # .env. index_chunks() unconditionally tries to write to
            # Postgres — without this, running the test suite silently wrote
            # this fixture chunk into the REAL production database every
            # time, permanently polluting real retrieval results with test
            # data (this is exactly how "doc.md" / "chunk_000" ended up
            # showing up in real --interactive sessions).
            index_path = index_chunks(chunks_file, output_dir, pgvector_dsn="postgresql://invalid:invalid@127.0.0.1:1/invalid")
            result = answer_question(index_path, "What does the remote work policy allow?", top_k=1, pgvector_dsn="postgresql://invalid:invalid@127.0.0.1:1/invalid")
            self.assertIn("policy", result["answer"].lower())
            self.assertIn("retrieved_chunks", result)
            eval_report = evaluate_rag(result)
            self.assertIn("retrieval_score", eval_report)
            self.assertIn("answer_generation_status", eval_report)


if __name__ == "__main__":
    unittest.main()
