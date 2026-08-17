import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.access_control import (
    ALL_DEPARTMENTS, filter_chunks_by_department, is_department_allowed,
)
from src.indexing.index_vectors import LocalVectorIndex
from src.retrieval.hybrid_search import hybrid_search


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


if __name__ == "__main__":
    unittest.main()
