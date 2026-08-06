"""
Compare classified_multilingual/ output against test_multilingual_samples/metadata.jsonl.

Run after ingestion + classification:
    python scripts/check_multilingual_accuracy.py
"""
import json
from pathlib import Path

GROUND_TRUTH = Path("data/samples/multilingual/metadata.jsonl")
CLASSIFIED_DIR = Path("data/classified")

FIELDS = [("true_department", "department"), ("true_doc_type", "doc_type"),
          ("true_language", "language"), ("true_sensitivity", "sensitivity")]


def main():
    truth = {}
    for line in GROUND_TRUTH.read_text().splitlines():
        row = json.loads(line)
        truth[row["filename"]] = row

    hits = {f[0]: 0 for f in FIELDS}
    total = 0

    for filename, expected in truth.items():
        stem = Path(filename).stem
        result_file = CLASSIFIED_DIR / f"{stem}.classified.json"
        if not result_file.exists():
            print(f"❌ MISSING: {result_file.name}")
            continue

        record = json.loads(result_file.read_text())
        c = record.get("classification", {})
        total += 1

        print(f"\n{filename}  (status: {c.get('status', c.get('classifier'))})")
        for truth_key, pred_key in FIELDS:
            exp = expected[truth_key]
            got = c.get(pred_key, "?")
            ok = str(got).upper() == str(exp).upper()
            hits[truth_key] += ok
            mark = "✅" if ok else "❌"
            print(f"  {mark} {pred_key:12s} expected={exp:12s} got={got}")

    print(f"\n=== Accuracy over {total} docs ===")
    for truth_key, _ in FIELDS:
        pct = (hits[truth_key] / total * 100) if total else 0
        print(f"  {truth_key.replace('true_', ''):12s}: {hits[truth_key]}/{total} ({pct:.0f}%)")


if __name__ == "__main__":
    main()