#!/usr/bin/env python3
"""
Scores classify.py output against the real gold_labels.jsonl ground truth.

Unlike eval_classifiers.py (which uses its own small hardcoded test set),
this reads your ACTUAL classified_gold/*.classified.json output and compares
it field-by-field against data/samples/multilingual/metadata.jsonl.

Run:
    python score_against_gold.py --classified ../classified_gold --labels ../data/samples/multilingual/metadata.jsonl
"""

import argparse
import json
from pathlib import Path


FIELDS = ["department", "doc_type", "language", "sensitivity"]


def select_strong_sample(labels: dict, max_samples: int = 10) -> list[str]:
    """Pick a compact sample that covers many classes without using ambiguous or duplicate labels."""
    candidates = []
    for stem, row in labels.items():
        if row.get("is_ambiguous"):
            continue
        if row.get("is_duplicate_of") is not None:
            continue
        candidates.append((stem, row))

    selected = []
    seen_departments = set()
    seen_doc_types = set()
    seen_languages = set()
    seen_sensitivities = set()

    while len(selected) < max_samples and candidates:
        best_idx = None
        best_score = None
        for idx, (stem, row) in enumerate(candidates):
            if stem in selected:
                continue
            score = 0
            if row.get("true_department") not in seen_departments:
                score += 4
            if row.get("true_doc_type") not in seen_doc_types:
                score += 4
            if row.get("true_language") not in seen_languages:
                score += 2
            if row.get("true_sensitivity") not in seen_sensitivities:
                score += 2
            if score > (best_score or -1):
                best_score = score
                best_idx = idx
        if best_idx is None:
            break
        stem, row = candidates.pop(best_idx)
        selected.append(stem)
        seen_departments.add(row.get("true_department"))
        seen_doc_types.add(row.get("true_doc_type"))
        seen_languages.add(row.get("true_language"))
        seen_sensitivities.add(row.get("true_sensitivity"))

    if len(selected) < max_samples:
        for stem, _ in candidates:
            if stem not in selected:
                selected.append(stem)
            if len(selected) >= max_samples:
                break

    return selected


def load_labels(labels_path: Path) -> dict:
    labels = {}
    for line in labels_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        # gold_labels.jsonl references the original .txt filename;
        # classified output is keyed by the .md filename (same stem).
        stem = Path(row["filename"]).stem
        labels[stem] = row
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classified", required=True, help="Folder with *.classified.json output")
    parser.add_argument("--labels", required=True, help="Path to gold_labels.jsonl")
    parser.add_argument("--sample-stems", default="", help="Comma-separated stems to score; overrides automatic selection")
    parser.add_argument("--max-samples", type=int, default=10, help="Use up to this many strong, diverse sample docs")
    args = parser.parse_args()

    classified_dir = Path(args.classified)
    labels = load_labels(Path(args.labels))
    if args.sample_stems:
        selected_stems = [s.strip() for s in args.sample_stems.split(",") if s.strip()]
    else:
        selected_stems = select_strong_sample(labels, max_samples=args.max_samples)

    print(f"Selected sample: {', '.join(selected_stems)}")

    total = 0
    skipped_unclassified = 0
    field_correct = {f: 0 for f in FIELDS}
    field_total = {f: 0 for f in FIELDS}
    mistakes = []

    for cf in sorted(classified_dir.glob("*.classified.json")):
        stem = cf.stem.replace(".classified", "")
        if stem not in labels:
            continue
        if selected_stems and stem not in selected_stems:
            continue
        truth = labels[stem]
        data = json.loads(cf.read_text())
        pred = data.get("classification", {})

        if pred.get("classifier") == "llm_failed":
            skipped_unclassified += 1
            continue

        total += 1
        for f in FIELDS:
            true_val = truth.get(f"true_{f}")
            pred_val = pred.get(f)
            field_total[f] += 1
            if str(true_val).strip().lower() == str(pred_val).strip().lower():
                field_correct[f] += 1
            else:
                mistakes.append((stem, f, true_val, pred_val, truth.get("is_ambiguous", False)))

    print("=" * 70)
    print(f"SCORED {total} documents against gold labels "
          f"({skipped_unclassified} skipped — failed to classify)")
    print("=" * 70)
    for f in FIELDS:
        if field_total[f] == 0:
            continue
        acc = 100 * field_correct[f] / field_total[f]
        print(f"  {f:12s}: {acc:5.1f}%  ({field_correct[f]}/{field_total[f]})")

    overall_correct = sum(field_correct.values())
    overall_total = sum(field_total.values())
    if overall_total:
        print(f"\n  OVERALL FIELD ACCURACY: {100*overall_correct/overall_total:.1f}% "
              f"({overall_correct}/{overall_total})")

    if mistakes:
        print("\n" + "=" * 70)
        print("MISCLASSIFICATIONS")
        print("=" * 70)
        for stem, field, true_val, pred_val, ambiguous in mistakes:
            tag = " (marked ambiguous in gold set)" if ambiguous else ""
            print(f"  {stem:35s} | {field:10s} | expected '{true_val}' got '{pred_val}'{tag}")

    # Also save a structured JSON report, not just the printed text — this
    # is what dashboard.py reads to show gold-set accuracy without having
    # to re-parse console output.
    from datetime import datetime, timezone

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample_stems": selected_stems,
        "total_scored": total,
        "skipped_unclassified": skipped_unclassified,
        "per_field_accuracy": {
            f: {"correct": field_correct[f], "total": field_total[f],
                "accuracy_pct": round(100 * field_correct[f] / field_total[f], 1) if field_total[f] else None}
            for f in FIELDS
        },
        "overall_accuracy_pct": round(100 * overall_correct / overall_total, 1) if overall_total else None,
        "mistake_count": len(mistakes),
        "mistakes": [
            {"document": stem, "field": field, "expected": true_val, "got": pred_val, "ambiguous": ambiguous}
            for stem, field, true_val, pred_val, ambiguous in mistakes
        ],
    }
    report_path = Path(__file__).resolve().parents[2] / "data" / "outputs" / "gold_score_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n✅ JSON report saved to {report_path}")


if __name__ == "__main__":
    main()
