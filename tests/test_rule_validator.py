import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.classification.rule_validator import validate_classification, validate_batch, MIN_CONFIDENCE_THRESHOLD


def _good_record(**overrides) -> dict:
    base = {
        "department": "HR",
        "doc_type": "Policy",
        "language": "EN",
        "sensitivity": "Internal",
        "confidence": 0.9,
        "classifier": "litellm",
    }
    base.update(overrides)
    return base


class RuleValidatorTests(unittest.TestCase):

    def test_valid_classification_passes_with_no_violations(self):
        result = validate_classification(_good_record())
        self.assertTrue(result.passed)
        self.assertFalse(result.needs_review)
        self.assertEqual(result.violations, [])

    def test_failed_classification_is_flagged_and_short_circuits(self):
        record = _good_record(department="UNCLASSIFIED", doc_type="UNCLASSIFIED",
                               language="UNCLASSIFIED", sensitivity="UNCLASSIFIED",
                               confidence=0.0, classifier="llm_failed")
        result = validate_classification(record)
        self.assertFalse(result.passed)
        self.assertTrue(result.needs_review)
        self.assertEqual(len(result.violations), 1)
        self.assertEqual(result.violations[0].rule, "classification_failed")

    def test_unknown_department_is_an_error(self):
        result = validate_classification(_good_record(department="Marketing"))
        self.assertFalse(result.passed)
        self.assertTrue(result.needs_review)
        rules = [v.rule for v in result.violations]
        self.assertIn("unknown_department", rules)

    def test_unknown_doc_type_is_an_error(self):
        result = validate_classification(_good_record(doc_type="Meeting Notes"))
        rules = [v.rule for v in result.violations]
        self.assertIn("unknown_doc_type", rules)
        self.assertTrue(result.needs_review)

    def test_unknown_sensitivity_is_an_error(self):
        result = validate_classification(_good_record(sensitivity="Top Secret"))
        rules = [v.rule for v in result.violations]
        self.assertIn("unknown_sensitivity", rules)
        self.assertTrue(result.needs_review)

    def test_unexpected_language_is_a_warning_not_blocking(self):
        result = validate_classification(_good_record(language="DE"))
        rules = [v.rule for v in result.violations]
        self.assertIn("unexpected_language", rules)
        # A warning alone must not set needs_review — only errors block.
        self.assertFalse(result.needs_review)
        self.assertFalse(result.passed)  # still "not clean", just not blocking

    def test_missing_confidence_is_an_error(self):
        result = validate_classification(_good_record(confidence=None))
        rules = [v.rule for v in result.violations]
        self.assertIn("missing_confidence", rules)
        self.assertTrue(result.needs_review)

    def test_low_confidence_is_a_warning_not_blocking(self):
        result = validate_classification(_good_record(confidence=MIN_CONFIDENCE_THRESHOLD - 0.01))
        rules = [v.rule for v in result.violations]
        self.assertIn("low_confidence", rules)
        self.assertFalse(result.needs_review)

    def test_confidence_exactly_at_threshold_does_not_flag(self):
        result = validate_classification(_good_record(confidence=MIN_CONFIDENCE_THRESHOLD))
        rules = [v.rule for v in result.violations]
        self.assertNotIn("low_confidence", rules)

    def test_contract_below_confidential_is_an_error(self):
        for bad_sensitivity in ("Public", "Internal"):
            with self.subTest(sensitivity=bad_sensitivity):
                result = validate_classification(_good_record(doc_type="Contract", sensitivity=bad_sensitivity, department="Legal"))
                rules = [v.rule for v in result.violations]
                self.assertIn("contract_under_classified", rules)
                self.assertTrue(result.needs_review)

    def test_contract_at_or_above_confidential_is_fine(self):
        for ok_sensitivity in ("Confidential", "Restricted"):
            with self.subTest(sensitivity=ok_sensitivity):
                result = validate_classification(_good_record(doc_type="Contract", sensitivity=ok_sensitivity, department="Legal"))
                rules = [v.rule for v in result.violations]
                self.assertNotIn("contract_under_classified", rules)

    def test_contract_rule_does_not_double_fire_with_unknown_sensitivity(self):
        # If sensitivity is itself invalid, the contract-sensitivity rule
        # shouldn't also fire on a value it can't meaningfully rank — the
        # unknown_sensitivity rule alone should cover that case.
        result = validate_classification(_good_record(doc_type="Contract", sensitivity="Top Secret"))
        rules = [v.rule for v in result.violations]
        self.assertIn("unknown_sensitivity", rules)
        self.assertNotIn("contract_under_classified", rules)

    def test_multiple_violations_all_reported_together(self):
        result = validate_classification(_good_record(department="Marketing", language="DE", confidence=0.2))
        rules = {v.rule for v in result.violations}
        self.assertEqual(rules, {"unknown_department", "unexpected_language", "low_confidence"})
        self.assertTrue(result.needs_review)  # unknown_department is an error

    def test_validate_batch_aggregates_correctly(self):
        records = [
            {"source_file": "a.md", **_good_record()},
            {"source_file": "b.md", **_good_record(department="Marketing")},
            {"source_file": "c.md", **_good_record(confidence=0.1)},  # warning only, still "passed"=False but not needs_review
        ]
        report = validate_batch(records)
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["passed"], 1)  # only record "a"
        self.assertEqual(report["needs_review"], 1)  # only record "b" (error-level)
        self.assertEqual(report["pass_rate"], round(1 / 3, 4))

    def test_validate_batch_handles_empty_input(self):
        report = validate_batch([])
        self.assertEqual(report["total"], 0)
        self.assertIsNone(report["pass_rate"])


if __name__ == "__main__":
    unittest.main()
