import tempfile
import unittest
from pathlib import Path

from ingestion.ingestion import discover_files, prepare_metadata_for_json


class DummyCallable:
    def __init__(self, value):
        self.value = value

    def get_title(self):
        return self.value


class IngestionTests(unittest.TestCase):
    def test_prepare_metadata_for_json_resolves_callable_values(self):
        dummy = DummyCallable("Example title")
        result = prepare_metadata_for_json({"title": dummy.get_title})
        self.assertEqual(result["title"], "Example title")

    def test_prepare_metadata_for_json_reports_field_name_for_non_serializable_values(self):
        with self.assertRaises(TypeError) as ctx:
            prepare_metadata_for_json({"title": object()})
        self.assertIn("title", str(ctx.exception))

    def test_discover_files_includes_text_documents(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = Path(tmpdir)
            (source_dir / "doc.txt").write_text("plain text", encoding="utf-8")
            (source_dir / "notes.md").write_text("markdown", encoding="utf-8")

            discovered = discover_files(source_dir)

            self.assertEqual([path.name for path in discovered], ["doc.txt"])


if __name__ == "__main__":
    unittest.main()
