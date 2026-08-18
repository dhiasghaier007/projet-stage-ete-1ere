#!/usr/bin/env python3
"""
fix_ground_truth.py — Corrects two labeling mistakes made when
build_wiki_corpus.py generated test_corpus_wiki/metadata.jsonl:

1. Sensitivity was hardcoded by department stereotype (HR->Confidential,
   Legal->Restricted, etc.) regardless of actual content. But the corpus
   content is generic Wikipedia prose (e.g. "what is a balance sheet"),
   which genuinely IS public information — it's not a real confidential
   HR file. Fix: set sensitivity to "Public" for all entries, since none
   of the scraped content contains actual sensitive company data.

2. The *_topics_*.csv files (bundled short-topic tables) were labeled with
   their department, but classify.py's own prompt has an explicit rule
   that a data table listing multiple department values isn't itself a
   department-specific document — it correctly classifies these as
   "General". Fix: relabel true_department to "General" for those files
   to match the classifier's intentional, documented behavior.

Writes a corrected copy rather than overwriting, so you can diff/compare.

Usage:
    python fix_ground_truth.py \\
        --input test_corpus_wiki/metadata.jsonl \\
        --output test_corpus_wiki/metadata_corrected.jsonl
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    fixed_sensitivity = 0
    fixed_department = 0
    total = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            total += 1

            if entry.get("true_sensitivity") != "Public":
                entry["true_sensitivity"] = "Public"
                fixed_sensitivity += 1

            filename = entry.get("filename", "")
            if "_topics_" in filename and filename.endswith(".csv"):
                if entry.get("true_department") != "General":
                    entry["true_department"] = "General"
                    fixed_department += 1

            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Processed {total} entries.")
    print(f"  Sensitivity corrected to 'Public': {fixed_sensitivity}")
    print(f"  Department corrected to 'General' (topic-bundle CSVs): {fixed_department}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
