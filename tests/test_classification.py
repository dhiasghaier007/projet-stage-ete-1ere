import os
import sys
import types
import unittest
from unittest.mock import patch

from src.classification.classify import classify_document_litellm, get_model_candidates


class LLMOnlyClassificationTests(unittest.TestCase):
    def test_get_model_candidates_reads_multiple_models_from_env(self):
        with patch.dict(os.environ, {"LITELLM_MODEL_NAMES": "ollama/mistral, gemini/gemini-2.0-flash"}, clear=True):
            self.assertEqual(
                get_model_candidates(),
                ["ollama/mistral", "gemini/gemini-2.0-flash"],
            )

    def test_get_model_candidates_tries_cloud_backends_before_ollama(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "valid-key", "OPENAI_API_KEY": "valid-key"}, clear=True):
            self.assertEqual(
                get_model_candidates(),
                ["gemini/gemini-3.5-flash-lite", "gpt-4o-mini", "ollama/qwen3.6"],
            )

    def test_get_model_candidates_keeps_ollama_last_when_explicitly_requested(self):
        with patch.dict(os.environ, {"LITELLM_MODEL_NAMES": "ollama/qwen3.6", "GEMINI_API_KEY": "valid-key", "OPENAI_API_KEY": "valid-key"}, clear=True):
            self.assertEqual(
                get_model_candidates(),
                ["gemini/gemini-3.5-flash-lite", "gpt-4o-mini", "ollama/qwen3.6"],
            )

    def test_classify_document_litellm_returns_unclassified_when_llm_errors(self):
        def fake_completion(*args, **kwargs):
            raise RuntimeError("AuthenticationError: invalid api key")

        fake_litellm = types.SimpleNamespace(completion=fake_completion)

        with patch.dict(os.environ, {"LITELLM_MODEL_NAMES": "ollama/mistral"}, clear=True):
            with patch.dict(sys.modules, {"litellm": fake_litellm}):
                result = classify_document_litellm("HR policy for remote work", "test.md")

        self.assertEqual(result["classifier"], "llm_failed")
        self.assertEqual(result["department"], "UNCLASSIFIED")
        self.assertEqual(result["doc_type"], "UNCLASSIFIED")
        self.assertEqual(result["language"], "UNCLASSIFIED")
        self.assertEqual(result["sensitivity"], "UNCLASSIFIED")
        self.assertEqual(result["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
