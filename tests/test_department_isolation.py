import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.access_control import (
    ALL_DEPARTMENTS, CANONICAL_DEPARTMENTS, filter_chunks_by_department,
    is_department_allowed, department_table_name, department_tables_for,
)
from src.indexing.index_vectors import LocalVectorIndex, index_chunks_by_department
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.postgres_hybrid import hybrid_search_pg


def _chunk(department, content="some content here about a topic"):
    return {"metadata": {"department": department}, "content": content}


class DepartmentIsolationUnitTests(unittest.TestCase):
    def test_all_departments_sentinel_allows_everything(self):
        self.assertTrue(is_department_allowed("Finance", ALL_DEPARTMENTS))
        self.assertTrue(is_department_allowed("Legal", ALL_DEPARTMENTS))

    def test_user_sees_their_own_department(self):
        self.assertTrue(is_department_allowed("HR", ["HR"]))

    def test_user_does_not_see_other_departments(self):
        self.assertFalse(is_department_allowed("Finance", ["HR"]))

    def test_user_can_belong_to_multiple_departments(self):
        self.assertTrue(is_department_allowed("Finance", ["HR", "Finance"]))
        self.assertFalse(is_department_allowed("Legal", ["HR", "Finance"]))

    def test_general_department_is_always_visible(self):
        # Company-wide content isn't isolated to any one department, even
        # for a user with a narrow, unrelated department list.
        self.assertTrue(is_department_allowed("General", ["Finance"]))

    def test_missing_or_empty_department_is_always_visible(self):
        self.assertTrue(is_department_allowed(None, ["Finance"]))
        self.assertTrue(is_department_allowed("", ["Finance"]))

    def test_filter_chunks_by_department_removes_disallowed_chunks(self):
        chunks = [_chunk("HR"), _chunk("Finance"), _chunk("General")]
        visible = filter_chunks_by_department(chunks, ["HR"])
        departments_seen = {c["metadata"]["department"] for c in visible}
        self.assertEqual(departments_seen, {"HR", "General"})


class DepartmentIsolationRetrievalTests(unittest.TestCase):
    """Proves department isolation actually changes what hybrid_search
    returns end to end — not just that the unit-level filter function
    works in isolation."""

    def setUp(self):
        self.index = LocalVectorIndex(dim=8)
        # Same embedding text on purpose so ranking differences are driven
        # by the department filter, not by which chunk is semantically closer.
        self.index.add_chunk("hr_chunk", "remote work policy details", {"department": "HR", "sensitivity": "Public"})
        self.index.add_chunk("finance_chunk", "remote work policy details", {"department": "Finance", "sensitivity": "Public"})
        self.index.add_chunk("general_chunk", "remote work policy details", {"department": "General", "sensitivity": "Public"})

    def test_unrestricted_by_default_sees_all_departments(self):
        results, _mode = hybrid_search(self.index, "remote work policy", top_k=5)
        seen = {r["metadata"]["department"] for r in results}
        self.assertEqual(seen, {"HR", "Finance", "General"})

    def test_department_scoped_caller_only_sees_their_department_and_general(self):
        results, _mode = hybrid_search(self.index, "remote work policy", top_k=5, departments=["HR"])
        seen = {r["metadata"]["department"] for r in results}
        self.assertEqual(seen, {"HR", "General"})
        self.assertNotIn("Finance", seen)


class DepartmentTableNamingTests(unittest.TestCase):
    """department_table_name is the one sanctioned place a department string
    becomes a SQL table identifier — these tests lock down both the happy
    path and the fail-loudly-on-anything-unrecognized behavior."""

    def test_canonical_department_maps_to_expected_table_name(self):
        self.assertEqual(department_table_name("HR"), "rag_chunks_hr")
        self.assertEqual(department_table_name("Finance"), "rag_chunks_finance")
        self.assertEqual(department_table_name("Legal"), "rag_chunks_legal")
        self.assertEqual(department_table_name("IT"), "rag_chunks_it")
        self.assertEqual(department_table_name("General"), "rag_chunks_general")

    def test_every_canonical_department_produces_a_safe_table_name(self):
        for department in CANONICAL_DEPARTMENTS:
            table_name = department_table_name(department)
            self.assertRegex(table_name, r"^[a-z0-9_]+$")

    def test_unrecognized_department_is_rejected_not_slugified(self):
        # A department the LLM invented (or an injection attempt smuggled in
        # as a "department") must be refused outright, not silently turned
        # into a table name — see the module docstring on why this is
        # intentionally strict rather than a general sanitizer.
        for bad_department in ["Marketing", "hr; DROP TABLE rag_chunks_hr;--", "", None, "HR "]:
            with self.assertRaises(ValueError):
                department_table_name(bad_department)


class DepartmentTablesForTests(unittest.TestCase):
    def test_all_departments_sentinel_returns_every_canonical_table(self):
        tables = department_tables_for(ALL_DEPARTMENTS)
        self.assertEqual(len(tables), len(CANONICAL_DEPARTMENTS))
        self.assertIn("rag_chunks_general", tables)

    def test_specific_department_list_includes_general_automatically(self):
        tables = department_tables_for(["HR"])
        self.assertIn("rag_chunks_hr", tables)
        self.assertIn("rag_chunks_general", tables)
        self.assertNotIn("rag_chunks_finance", tables)

    def test_department_list_is_deduplicated(self):
        tables = department_tables_for(["HR", "HR", "General"])
        self.assertEqual(tables.count("rag_chunks_hr"), 1)
        self.assertEqual(tables.count("rag_chunks_general"), 1)

    def test_empty_department_list_still_gets_general(self):
        tables = department_tables_for([])
        self.assertEqual(tables, ["rag_chunks_general"])


class IndexChunksByDepartmentTests(unittest.TestCase):
    """index_chunks_by_department is the real storage-layer implementation
    of the 'Two Department Knowledge Bases' deliverable — one local index
    file and one pgvector table per department, never a shared file that
    departments could silently overwrite each other in."""

    def _write_chunks_file(self, tmp_path: Path, records) -> Path:
        chunks_file = tmp_path / "chunks.jsonl"
        with chunks_file.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return chunks_file

    def test_creates_one_local_index_file_per_department(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            chunks_file = self._write_chunks_file(tmp_path, [
                {"chunk_id": "c1", "department": "HR", "content": "remote work policy", "source_file": "a.md", "created_at": "now"},
                {"chunk_id": "c2", "department": "Finance", "content": "invoice terms", "source_file": "b.md", "created_at": "now"},
                {"chunk_id": "c3", "department": "HR", "content": "vacation policy", "source_file": "a.md", "created_at": "now"},
            ])
            output_dir = tmp_path / "out"

            # Unreachable DSN — same pattern as test_retrieval.py's fixture
            # isolation, so this never touches a real database.
            index_files = index_chunks_by_department(
                chunks_file, output_dir, pgvector_dsn="postgresql://invalid:invalid@127.0.0.1:1/invalid"
            )

            self.assertIn("HR", index_files)
            self.assertIn("Finance", index_files)
            self.assertNotIn("Legal", index_files)  # no Legal chunks in this batch

            hr_payload = json.loads(index_files["HR"].read_text(encoding="utf-8"))
            finance_payload = json.loads(index_files["Finance"].read_text(encoding="utf-8"))

            self.assertEqual(set(hr_payload["ids"]), {"c1", "c3"})
            self.assertEqual(set(finance_payload["ids"]), {"c2"})

            # Files must be distinctly named, not a shared "local_index.json"
            # that one department's write would silently clobber.
            self.assertNotEqual(index_files["HR"].name, index_files["Finance"].name)

    def test_writes_a_manifest_describing_every_department_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            chunks_file = self._write_chunks_file(tmp_path, [
                {"chunk_id": "c1", "department": "IT", "content": "server maintenance window", "source_file": "a.md", "created_at": "now"},
            ])
            output_dir = tmp_path / "out"
            index_chunks_by_department(chunks_file, output_dir, pgvector_dsn="postgresql://invalid:invalid@127.0.0.1:1/invalid")

            manifest = json.loads((output_dir / "department_index_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("IT", manifest["departments"])
            self.assertEqual(manifest["departments"]["IT"]["table_name"], "rag_chunks_it")
            self.assertEqual(manifest["departments"]["IT"]["chunk_count"], 1)

    def test_unrecognized_department_chunks_are_skipped_not_silently_dumped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            chunks_file = self._write_chunks_file(tmp_path, [
                {"chunk_id": "c1", "department": "Marketing", "content": "campaign brief", "source_file": "a.md", "created_at": "now"},
                {"chunk_id": "c2", "department": "HR", "content": "onboarding checklist", "source_file": "b.md", "created_at": "now"},
            ])
            output_dir = tmp_path / "out"
            index_files = index_chunks_by_department(chunks_file, output_dir, pgvector_dsn="postgresql://invalid:invalid@127.0.0.1:1/invalid")

            # The unrecognized "Marketing" chunk must not end up in ANY
            # department's index (not silently merged into General/HR).
            self.assertNotIn("Marketing", index_files)
            self.assertIn("HR", index_files)
            hr_payload = json.loads(index_files["HR"].read_text(encoding="utf-8"))
            self.assertEqual(hr_payload["ids"], ["c2"])

            manifest = json.loads((output_dir / "department_index_manifest.json").read_text(encoding="utf-8"))
            self.assertIn("Marketing", manifest["unknown_departments_skipped"])


class ByDepartmentRetrievalDegradationTests(unittest.TestCase):
    """hybrid_search_pg(by_department=True) must degrade exactly like the
    existing single-table path when Postgres isn't reachable — returning
    None, never raising and never silently returning an empty result that
    could be mistaken for 'genuinely no matches'."""

    def test_by_department_returns_none_without_reachable_postgres(self):
        result = hybrid_search_pg(
            "any question",
            dsn="postgresql://nobody:nobody@localhost:1/nonexistent",
            by_department=True,
            departments=["HR"],
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
