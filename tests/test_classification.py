import os
import sys
import types
import unittest
from unittest.mock import patch

from classification.classify import classify_document_litellm, get_model_candidates


class LLMOnlyClassificationTests(unittest.TestCase):
    def test_get_model_candidates_reads_multiple_models_from_env(self):
        with patch.dict(os.environ, {"LITELLM_MODEL_NAMES": "ollama/mistral, gemini/gemini-2.0-flash"}, clear=True):
            self.assertEqual(
                get_model_candidates(),
                ["ollama/mistral", "gemini/gemini-2.0-flash"],
            )

    def test_classify_document_litellm_falls_back_to_heuristic_when_llm_errors(self):
        def fake_completion(*args, **kwargs):
            raise RuntimeError("AuthenticationError: invalid api key")

        fake_litellm = types.SimpleNamespace(completion=fake_completion)

        with patch.dict(os.environ, {"LITELLM_MODEL_NAMES": "ollama/mistral"}, clear=True):
            with patch.dict(sys.modules, {"litellm": fake_litellm}):
                result = classify_document_litellm("HR policy for remote work", "test.md")

        self.assertEqual(result["classifier"], "heuristic_fallback")
        self.assertEqual(result["department"], "HR")
        self.assertEqual(result["doc_type"], "Policy")
        self.assertEqual(result["language"], "EN")
        self.assertGreaterEqual(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
