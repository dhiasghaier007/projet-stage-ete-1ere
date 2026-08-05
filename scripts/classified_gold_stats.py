#!/usr/bin/env python3

import json
from pathlib import Path


def load_classified_records(directory: Path) -> list[dict]:
    records = []
    for meta_file in sorted(directory.glob("*.classified.json")):
        with meta_file.open("r", encoding="utf-8") as f:
            records.append(json.load(f))
    return records


def summarize(records: list[dict]) -> dict:
    counts = {
        "total_documents": len(records),
        "departments": {},
        "doc_types": {},
        "sensitivities": {},
    }
    for record in records:
        classification = record.get("classification", {})
        for field in ("department", "doc_type", "sensitivity"):
            value = classification.get(field) or "Unknown"
            counts_field = counts[f"{field}s"]
            counts_field[value] = counts_field.get(value, 0) + 1
    return counts


def format_summary(counts: dict) -> str:
    lines = [
        f"Total classified documents: {counts['total_documents']}",
        "",
        "Departments:",
    ]
    for department, total in sorted(counts["departments"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {department}: {total}")
    lines.append("")
    lines.append("Document types:")
    for doc_type, total in sorted(counts["doc_types"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {doc_type}: {total}")
    lines.append("")
    lines.append("Sensitivities:")
    for sensitivity, total in sorted(counts["sensitivities"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {sensitivity}: {total}")
    return "\n".join(lines)


def main() -> None:
    directory = Path("classified_gold")
    if not directory.is_dir():
        print(f"Directory not found: {directory}")
        raise SystemExit(1)

    records = load_classified_records(directory)
    summary = summarize(records)
    print(format_summary(summary))


if __name__ == "__main__":
    main()
