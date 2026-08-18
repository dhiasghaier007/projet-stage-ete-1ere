"""
Rule-based validation — sits right after LLM classification, before a
document is trusted enough to move on to chunking/indexing.

Why this exists: classify.py's LLM step has no silent fallback for outright
failure (an unreachable LLM produces an explicit "UNCLASSIFIED" record, not
a guess) — but it has no defense at all against a LLM call that *succeeds*
while returning something implausible: a legal contract tagged "Public", a
department name the LLM invented that doesn't exist in this company, a
confidence score of 0.97 on a document the classifier clearly misread. LLMs
are fluent even when wrong, and a wrong-but-confident classification is
exactly the kind of error that's invisible until it causes real harm (e.g.
a Restricted document indexed as Public and freely retrievable).

This module runs a fixed set of deterministic rules against a completed
classification and reports every violation explicitly — it never silently
"fixes" a bad classification by guessing a better one; that would just be a
second unaudited guess stacked on the first. Documents that fail a
hard ("error"-severity) rule are flagged as needing_review=True and should
not proceed to chunking until a human confirms them — see the Classification
Rulebook (RULEBOOK.md) for what each rule checks and why.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

from src.retrieval.access_control import SENSITIVITY_LEVELS, CANONICAL_DEPARTMENTS as _AC_CANONICAL_DEPARTMENTS

# Canonical value sets. These intentionally mirror the exact options given
# to the LLM in classify.py's prompt — a value outside this set means the
# LLM either hallucinated a new category or the prompt/rules have drifted
# out of sync with each other, both worth catching.
#
# CANONICAL_DEPARTMENTS is re-exported from access_control.py rather than
# kept as its own copy here — access_control.py is now the single source of
# truth (department names also flow into pgvector table names there, via
# department_table_name()), so classification's validation rules and
# storage's table naming can never silently drift onto two different ideas
# of what a valid department is.
CANONICAL_DEPARTMENTS = _AC_CANONICAL_DEPARTMENTS
CANONICAL_DOC_TYPES = {"Policy", "Invoice", "Report", "Contract", "Data Table", "Document", "Email"}
CANONICAL_SENSITIVITY = set(SENSITIVITY_LEVELS.keys())  # Public, Internal, Confidential, Restricted
CANONICAL_LANGUAGES = {"EN", "FR", "AR"}  # the corpus's actual expected languages (see detect_language)

# Below this self-reported confidence, a classification is flagged for
# human review even if every other rule passes — a low-confidence LLM
# guess shouldn't silently carry the same trust as a confident one.
MIN_CONFIDENCE_THRESHOLD = 0.5

# doc_type values that are inherently sensitive regardless of what the LLM
# guessed for sensitivity — a rule catches these being under-classified.
_INHERENTLY_SENSITIVE_DOC_TYPES = {"Contract"}
_MIN_SENSITIVITY_FOR_CONTRACTS = "Confidential"  # Contract must be at least this sensitive


@dataclass
class Violation:
    rule: str
    severity: str  # "error" (blocks — needs_review=True) or "warning" (flagged, not blocking)
    message: str


@dataclass
class ValidationResult:
    passed: bool
    needs_review: bool
    violations: List[Violation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "needs_review": self.needs_review,
            "violations": [{"rule": v.rule, "severity": v.severity, "message": v.message} for v in self.violations],
        }


def _rank(label: str) -> int:
    """Sensitivity rank, fail-closed (unrecognized label ranks as most
    restrictive) — same policy as access_control.is_allowed, so a
    validation rule can't be tricked by an unrecognized label into thinking
    a document is LESS sensitive than it might actually be."""
    return SENSITIVITY_LEVELS.get(label, max(SENSITIVITY_LEVELS.values()))


def validate_classification(classification: Dict[str, Any]) -> ValidationResult:
    """Run every rule against one classification dict (the same shape
    classify_document_litellm() returns / ClassificationRecord holds:
    department, doc_type, language, sensitivity, confidence, classifier,
    error). Returns a ValidationResult — never mutates or "corrects" the
    input, only reports what's wrong with it."""
    violations: List[Violation] = []

    department = classification.get("department")
    doc_type = classification.get("doc_type")
    language = classification.get("language")
    sensitivity = classification.get("sensitivity")
    confidence = classification.get("confidence")
    classifier_failed = classification.get("classifier") == "llm_failed" or department == "UNCLASSIFIED"

    # Rule: classification failed outright (LLM unreachable/denied) — this
    # is already surfaced elsewhere in classify.py's summary counts, but a
    # validator that silently passed an UNCLASSIFIED record would defeat
    # the whole point of "no silent fallback" — it must show up here too.
    if classifier_failed:
        violations.append(Violation(
            "classification_failed", "error",
            "Document has no real classification (LLM call failed or was denied) — "
            "must not proceed to chunking/indexing until reclassified.",
        ))
        # No other rule can meaningfully evaluate an UNCLASSIFIED record —
        # every field below is a placeholder, not a real value.
        return ValidationResult(passed=False, needs_review=True, violations=violations)

    # Rule: department must be a known company department, not something
    # the LLM invented (e.g. "Marketing" if that's not a real dept here).
    if department not in CANONICAL_DEPARTMENTS:
        violations.append(Violation(
            "unknown_department", "error",
            f"Department '{department}' is not one of the recognized departments "
            f"({', '.join(sorted(CANONICAL_DEPARTMENTS))}).",
        ))

    # Rule: doc_type must be one of the recognized categories.
    if doc_type not in CANONICAL_DOC_TYPES:
        violations.append(Violation(
            "unknown_doc_type", "error",
            f"Document type '{doc_type}' is not one of the recognized types "
            f"({', '.join(sorted(CANONICAL_DOC_TYPES))}).",
        ))

    # Rule: sensitivity must be one of the four canonical labels — this is
    # especially important since access_control.py fails closed on an
    # unrecognized label anyway, so an invalid label here silently makes a
    # document maximally restricted at retrieval time without anyone
    # having decided that on purpose.
    if sensitivity not in CANONICAL_SENSITIVITY:
        violations.append(Violation(
            "unknown_sensitivity", "error",
            f"Sensitivity '{sensitivity}' is not one of {sorted(CANONICAL_SENSITIVITY, key=lambda s: SENSITIVITY_LEVELS[s])}. "
            f"This document will be treated as maximally restricted at retrieval time as a result.",
        ))

    # Rule: language must be one of the corpus's expected languages.
    if language not in CANONICAL_LANGUAGES:
        violations.append(Violation(
            "unexpected_language", "warning",
            f"Language '{language}' is outside the expected set {sorted(CANONICAL_LANGUAGES)} — "
            f"either a genuinely new language appeared in the corpus, or the classifier misread the document.",
        ))

    # Rule: confidence must be present and above the review threshold.
    if not isinstance(confidence, (int, float)):
        violations.append(Violation(
            "missing_confidence", "error",
            "Classification has no usable confidence score.",
        ))
    elif confidence < MIN_CONFIDENCE_THRESHOLD:
        violations.append(Violation(
            "low_confidence", "warning",
            f"Confidence {confidence:.2f} is below the review threshold ({MIN_CONFIDENCE_THRESHOLD}) — "
            f"recommend human review before trusting this classification.",
        ))

    # Rule: legally binding documents must never be under-classified.
    # A contract tagged Public or Internal is a real business risk, not a
    # minor labeling slip — this is an "error", not a "warning".
    if doc_type in _INHERENTLY_SENSITIVE_DOC_TYPES and sensitivity in CANONICAL_SENSITIVITY:
        if _rank(sensitivity) < _rank(_MIN_SENSITIVITY_FOR_CONTRACTS):
            violations.append(Violation(
                "contract_under_classified", "error",
                f"Document type '{doc_type}' was classified as '{sensitivity}', but contracts/legal "
                f"agreements must be at least '{_MIN_SENSITIVITY_FOR_CONTRACTS}'.",
            ))

    has_error = any(v.severity == "error" for v in violations)
    return ValidationResult(passed=not violations, needs_review=has_error, violations=violations)


def validate_batch(classified_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a full batch of classification dicts (e.g. the "records"
    list from classify.py's summary metadata) and produce an aggregate
    report — used by the standalone re-validation CLI and, eventually, the
    quality dashboard."""
    results = []
    needs_review_count = 0
    passed_count = 0

    for record in classified_records:
        result = validate_classification(record)
        if result.passed:
            passed_count += 1
        if result.needs_review:
            needs_review_count += 1
        results.append({
            "source_file": record.get("source_file", "unknown"),
            **result.to_dict(),
        })

    total = len(classified_records)
    return {
        "total": total,
        "passed": passed_count,
        "needs_review": needs_review_count,
        "pass_rate": round(passed_count / total, 4) if total else None,
        "results": results,
    }


def main() -> None:
    """Standalone re-validation: run rule-based validation against an
    already-classified batch (classify.py's --metadata output), without
    needing to re-run any LLM classification. Useful for checking older
    runs against updated rules, or generating a review report on demand.
    """
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description="Re-run rule-based validation against a classify.py summary file (classified_metadata.json)."
    )
    parser.add_argument("--metadata", required=True, help="Path to classify.py's --metadata output (has a 'records' list)")
    parser.add_argument("--output", default=None, help="Where to write the validation report JSON (default: <metadata>.validation.json)")
    args = parser.parse_args()

    metadata_path = Path(args.metadata)
    summary = _json.loads(metadata_path.read_text())
    records = summary.get("records", [])

    report = validate_batch(records)

    output_path = Path(args.output) if args.output else metadata_path.with_suffix(".validation.json")
    output_path.write_text(_json.dumps(report, indent=2))

    print(f"Validated {report['total']} classification(s): {report['passed']} passed, "
          f"{report['needs_review']} flagged for review.")
    if report["needs_review"] > 0:
        print("\nFlagged documents:")
        for item in report["results"]:
            if item["needs_review"]:
                reasons = "; ".join(v["message"] for v in item["violations"] if v["severity"] == "error")
                print(f"  🚩 {item['source_file']}: {reasons}")
    print(f"\nFull report written to {output_path}")


if __name__ == "__main__":
    main()
