#!/usr/bin/env python3
"""
eval_corpus.py — General-purpose classification accuracy evaluator.

Unlike eval_classifiers.py (which tests 8 hardcoded English documents baked
into the script), this evaluates the LIVE classifier against ANY labeled
dataset you point it at — your Wikipedia test corpus today, real company
documents tomorrow. Nothing here is hardcoded to a specific dataset.

GROUND TRUTH FORMAT (the standard to reuse for any future labeled set):
A JSONL file where each line is a JSON object with:
    {
      "filename": "original_source_filename.ext",   # required — matches the
                                                      # file BEFORE ingestion
                                                      # (e.g. "report.pdf",
                                                      # not the ingested .md)
      "true_department": "HR",                        # optional
      "true_doc_type": "Policy",                       # optional
      "true_language": "EN",                           # optional
      "true_sensitivity": "Internal"                   # optional
    }
Any subset of the true_* fields may be present — only fields that exist in
a given line are scored for that document. This is exactly the schema
build_wiki_corpus.py already writes to metadata.jsonl, so it works out of
the box for that corpus, and any team preparing a new labeled set for a
different document type can follow the same convention.

Matching logic: ground truth "filename" (pre-ingestion, e.g. "report.pdf")
is matched to the ingested "{stem}.md" in --processed (Stage 1 output),
since that's the naming convention ingestion.py already uses.

Note on doc_type: your corpus's ground-truth doc_type labels (e.g.
"Article", "DataSheet") and the classifier's own taxonomy (e.g. "Document",
"Data Table") come from different vocabularies, so doc_type accuracy here
is informational, not a strict pass/fail signal — department, language, and
sensitivity are the fields worth trusting for a hard number.

Usage:
    python eval_corpus.py \\
        --processed data/processed \\
        --ground-truth test_corpus_wiki/metadata.jsonl \\
        --output classification_eval_corpus_report

Options:
    --limit N       Only evaluate the first N matched documents (quick smoke test)
    --verbose        Print each document's expected vs actual, not just the summary
    --delay SECONDS  Override the between-call delay (default: same as classify.py's
                     CLASSIFY_REQUEST_DELAY, respects free-tier rate limits)
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Reuse the exact same classification function the real pipeline uses —
# this is what makes the eval trustworthy: it's testing the live code path,
# not a reimplementation of it.
# classify.py contains an internal absolute import
# ("from src.classification.rule_validator import ...") that only resolves
# if the repo root is on sys.path — so we add both the repo root (three
# levels up from this file: src/classification/eval_corpus.py -> repo root)
# and this file's own directory, making this script runnable directly
# (`python3 eval_corpus.py`) without needing `-m` module invocation.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]  # src/classification -> src -> repo root
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))
try:
    from classify import classify_document_litellm, REQUEST_DELAY_SECONDS
except ImportError:
    print("ERROR: could not import classify.py. Make sure eval_corpus.py "
          f"sits in src/classification/ alongside classify.py. "
          f"(Tried repo root: {_REPO_ROOT})", file=sys.stderr)
    raise

SCORED_FIELDS = ["department", "doc_type", "language", "sensitivity"]

# doc_type ground truth (Article/Report/Reference/DataSheet, from
# build_wiki_corpus.py's format-based labeling) and the classifier's own
# taxonomy (Document/Data Table/Policy/Invoice/etc., from its prompt) are
# different vocabularies describing overlapping concepts, not disagreements.
# This maps ground-truth labels to the set of classifier outputs that count
# as equivalent, so doc_type scoring reflects real mismatches instead of
# vocabulary mismatches. Extend this as you add more ground-truth label
# types or observe new classifier output categories.
#
# IMPORTANT CONTEXT for the entries below: classify.py's prompt only ever
# offers the LLM these 7 doc_type options: Policy, Invoice, Report,
# Contract, Data Table, Document, Email — see CANONICAL_DOC_TYPES in
# rule_validator.py. Ground-truth labels like "Memo", "Announcement",
# "Press Release", "Statement", "Blog Post", "Procedure" (introduced by the
# sensitivity_corpus test set) were NEVER options the classifier could
# choose — this is a genuine taxonomy gap between the ground truth and the
# classifier's prompt, not a case of the classifier guessing wrong among
# available choices. Mapping these to "document" (the correct generic
# catch-all given the classifier's real taxonomy) reflects the classifier
# doing the best it could with the categories it was actually given. If
# accurate doc_type detection for real business document types like these
# matters going forward, the real fix is expanding CANONICAL_DOC_TYPES /
# the classify.py prompt itself — not just this eval mapping.
DOC_TYPE_EQUIVALENTS = {
    "article": {"document", "report", "article"},
    "report": {"document", "report"},
    "reference": {"document", "reference"},
    "datasheet": {"data table", "datasheet", "spreadsheet"},
    "announcement": {"document"},
    "memo": {"document"},  # deliberately NOT including "email" — see note below
    "press release": {"document", "report"},
    "statement": {"document"},
    "blog post": {"document"},
    "procedure": {"document"},
}
# Note on "memo": sens_06_finance_internal_budget.txt was labeled Memo but
# classified as Email. We deliberately do NOT equivalence-match memo↔email
# here — a memo classified as an email is a real, worth-noticing distinction
# (format, not just vocabulary), not vocabulary noise like the others above.
# If you inspect that document and confirm it's genuinely formatted as an
# email (memos are often distributed via email in practice), add "email" to
# the memo set above; until then, leaving it unmapped keeps it visible as a
# flagged mismatch rather than silently hiding it.


def load_ground_truth(path: Path) -> dict:
    """Returns {filename: {field: expected_value, ...}, ...}"""
    entries = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARNING: skipping malformed line {line_no} in {path}: {e}")
                continue
            fname = obj.get("filename")
            if not fname:
                print(f"WARNING: skipping line {line_no} — no 'filename' field")
                continue
            expected = {}
            for field in SCORED_FIELDS:
                key = f"true_{field}"
                if key in obj:
                    expected[field] = obj[key]
            entries[fname] = expected
    return entries


def match_to_processed(ground_truth: dict, processed_dir: Path) -> list:
    """Match each ground-truth filename (pre-ingestion) to its processed .md file."""
    matched = []
    unmatched = []
    for orig_filename, expected in ground_truth.items():
        stem = Path(orig_filename).stem
        md_path = processed_dir / f"{stem}.md"
        if md_path.exists():
            matched.append((orig_filename, md_path, expected))
        else:
            unmatched.append(orig_filename)
    return matched, unmatched


def is_match(field: str, expected, actual) -> bool:
    exp_norm = normalize(expected)
    act_norm = normalize(actual)
    if exp_norm == act_norm:
        return True
    if field == "doc_type" and exp_norm in DOC_TYPE_EQUIVALENTS:
        return act_norm in DOC_TYPE_EQUIVALENTS[exp_norm]
    return False


def normalize(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def evaluate(matched: list, verbose: bool, delay: float) -> dict:
    field_correct = defaultdict(int)
    field_total = defaultdict(int)
    mismatches = defaultdict(list)  # field -> [(filename, expected, actual), ...]
    per_doc_results = []
    failed = 0

    for i, (orig_filename, md_path, expected) in enumerate(matched, 1):
        content = md_path.read_text(encoding="utf-8")
        result = classify_document_litellm(content, md_path.name)

        if result.get("classifier") == "llm_failed" or result.get("department") == "UNCLASSIFIED":
            failed += 1

        doc_result = {"filename": orig_filename, "expected": expected, "actual": {}}
        for field in SCORED_FIELDS:
            if field not in expected:
                continue
            actual_val = result.get(field, "")
            doc_result["actual"][field] = actual_val
            field_total[field] += 1
            if is_match(field, expected[field], actual_val):
                field_correct[field] += 1
            else:
                mismatches[field].append((orig_filename, expected[field], actual_val))

        per_doc_results.append(doc_result)

        if verbose:
            status_bits = []
            for field in SCORED_FIELDS:
                if field not in expected:
                    continue
                mark = "✅" if is_match(field, expected[field], doc_result["actual"][field]) else "❌"
                status_bits.append(f"{field}:{mark}")
            print(f"  [{i}/{len(matched)}] {orig_filename:55s} {' '.join(status_bits)}")

        time.sleep(delay)

    return {
        "field_correct": dict(field_correct),
        "field_total": dict(field_total),
        "mismatches": {k: v for k, v in mismatches.items()},
        "per_doc_results": per_doc_results,
        "failed": failed,
        "total_docs": len(matched),
    }


def format_report(results: dict) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("CLASSIFICATION ACCURACY — CORPUS EVALUATION")
    lines.append("=" * 70)
    lines.append(f"Documents evaluated: {results['total_docs']}")
    lines.append(f"Classifier failures (UNCLASSIFIED / llm_failed): {results['failed']}")
    lines.append("")
    lines.append("Per-field accuracy:")
    for field in SCORED_FIELDS:
        correct = results["field_correct"].get(field, 0)
        total = results["field_total"].get(field, 0)
        if total == 0:
            lines.append(f"  {field:12s}: no ground-truth labels provided, skipped")
            continue
        pct = 100 * correct / total
        note = ""
        lines.append(f"  {field:12s}: {correct}/{total}  ({pct:.1f}%){note}")
    lines.append("")
    for field, mismatch_list in results["mismatches"].items():
        if not mismatch_list:
            continue
        lines.append(f"Mismatches — {field}:")
        for filename, expected, actual in mismatch_list:
            lines.append(f"  {filename:50s} expected={expected!r:20s} got={actual!r}")
        lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", required=True, help="Folder with Stage 1 outputs (.md files)")
    parser.add_argument("--ground-truth", required=True, help="Path to ground-truth JSONL (see docstring for schema)")
    parser.add_argument("--output", default="classification_eval_corpus_report", help="Report filename prefix (writes .txt and .json)")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate first N matched documents")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--delay", type=float, default=None, help="Seconds between LLM calls (default: same as classify.py)")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    gt_path = Path(args.ground_truth)
    delay = args.delay if args.delay is not None else REQUEST_DELAY_SECONDS

    if not processed_dir.is_dir():
        print(f"ERROR: --processed dir not found: {processed_dir}", file=sys.stderr)
        sys.exit(1)
    if not gt_path.is_file():
        print(f"ERROR: --ground-truth file not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    ground_truth = load_ground_truth(gt_path)
    print(f"Loaded {len(ground_truth)} ground-truth entries from {gt_path}")

    matched, unmatched = match_to_processed(ground_truth, processed_dir)
    if unmatched:
        print(f"WARNING: {len(unmatched)} ground-truth entries had no matching "
              f"processed file (not yet ingested?):")
        for u in unmatched[:10]:
            print(f"    - {u}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")

    if args.limit:
        matched = matched[: args.limit]

    if not matched:
        print("ERROR: no ground-truth entries matched any processed file. "
              "Check --processed points at the right ingestion output, and "
              "that ground-truth filenames match pre-ingestion source names.")
        sys.exit(1)

    print(f"Evaluating {len(matched)} matched documents (delay={delay}s between calls)...\n")

    results = evaluate(matched, verbose=args.verbose, delay=delay)
    report = format_report(results)
    print("\n" + report)

    txt_path = Path(f"{args.output}.txt")
    txt_path.write_text(report, encoding="utf-8")
    print(f"\n✅ Text report saved to {txt_path}")

    json_path = Path(f"{args.output}.json")
    json_payload = {
        **results,
        "ground_truth_file": str(gt_path),
        "processed_dir": str(processed_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_accuracy_pct": {
            field: round(100 * results["field_correct"].get(field, 0) / results["field_total"][field], 1)
            for field in SCORED_FIELDS if results["field_total"].get(field, 0) > 0
        },
    }
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ JSON report saved to {json_path}")


if __name__ == "__main__":
    main()
