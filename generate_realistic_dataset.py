"""
Improved mock dataset generator — builds a more realistic, harder test corpus
than simple template fill-ins.

Key differences from the basic version:
  1. Splits output into dev/ (tune freely) and gold/ (never look at while tuning)
  2. Injects deliberately ambiguous cross-category documents
  3. Injects exact + near-duplicates (to test Stage 3 dedup later)
  4. Injects OCR-style noise on a fraction of documents
  5. Writes REAL files (.txt/.csv/.html) into folders your actual ingestion
     pipeline can read directly — not a separate JSONL sandbox

Usage:
  python3 generate_realistic_dataset.py --dev_count 30 --gold_count 60 --output_dir ./test_corpus

Output structure:
  test_corpus/
    dev/            <- documents + dev_labels.jsonl (look at these freely)
    gold/           <- documents + gold_labels.jsonl (DO NOT peek while tuning)
"""

import json
import random
import argparse
import os
from pathlib import Path

DEPARTMENTS = ["HR", "Finance", "Legal", "IT", "General"]
DOC_TYPES = ["Policy", "Invoice", "Report", "Contract", "Email"]
LANGUAGES = ["EN", "FR", "AR"]
SENSITIVITY_MAP = {"HR": "Confidential", "Finance": "Internal", "Legal": "Restricted",
                    "IT": "Internal", "General": "Public"}

# ---- Base content templates, WITH VARIATION per department/type ----
# Multiple phrasing variants per combo so it's not the exact same sentence every time.
TEMPLATES = {
    ("HR", "Policy"): [
        "Remote Work Policy\n\nEmployees in {dept} may work remotely up to {n} days weekly, pending manager approval. Effective {date}.",
        "Leave and Absence Guidelines\n\nThis document outlines the {dept} department's approach to requesting time off, sick leave, and parental leave, effective {date}.",
        "Code of Conduct — {dept} Department\n\nAll staff must adhere to the following standards of professional behavior, reviewed on {date}.",
    ],
    ("Finance", "Invoice"): [
        "INVOICE #{n}\n\nBill To: Acme Corp\nDescription: Consulting services for Q{n} review.\nAmount Due: ${amount}\nDue: {date}",
        "PAYMENT REQUEST #{n}\n\nVendor: Northwind Supplies\nFor: Office equipment purchase order #{n}.\nTotal: ${amount}\nDue: {date}",
    ],
    ("Legal", "Contract"): [
        "SERVICE AGREEMENT\n\nBetween Company and Vendor {n}, effective {date}. See Section {n} on confidentiality and termination.",
        "NON-DISCLOSURE AGREEMENT\n\nEntered into by Company and Partner {n} as of {date}, governing exchange of confidential information.",
    ],
    ("IT", "Report"): [
        "System Uptime Report — Week {n}\n\nOverall uptime: 99.{n}%. {n} incidents recorded, resolved within SLA.",
        "Security Audit Summary #{n}\n\n{n} vulnerabilities identified during the {date} scan; {n} already patched.",
    ],
    ("General", "Email"): [
        "Subject: Team Lunch\n\nHi all, team lunch on Friday at {n}pm. RSVP by {date}.",
        "Subject: Office Announcement\n\nPlease note the office will close early on {date} for maintenance.",
    ],
}

# Deliberately AMBIGUOUS documents — designed to confuse a naive classifier.
# These test whether your classifier does more than keyword matching.
AMBIGUOUS_TEMPLATES = [
    {"content": "INVOICE #{n}\n\nBill To: Acme Corp\nDescription: Legal consulting and policy drafting services for HR compliance review.\nAmount Due: ${amount}",
     "true_department": "Finance", "true_doc_type": "Invoice"},  # mentions "policy", "HR", "Legal" but IS an invoice
    {"content": "MEMO\n\nRe: Updated IT Security Policy affecting Finance department server access.\nEffective {date}, all Finance staff must use two-factor authentication.",
     "true_department": "IT", "true_doc_type": "Policy"},  # mentions Finance but is an IT policy
    {"content": "CONTRACT AMENDMENT — Invoice Payment Terms\n\nThis amendment to the original service agreement adjusts invoice payment terms from 30 to 60 days.",
     "true_department": "Legal", "true_doc_type": "Contract"},  # mentions "Invoice" but is a legal contract
]


def inject_ocr_noise(text: str, intensity=0.02) -> str:
    """Randomly corrupt a small fraction of characters to simulate scanned/OCR'd documents."""
    chars = list(text)
    glitch_map = {"o": "0", "l": "1", "e": "3", "a": "@", "s": "5"}
    for i, c in enumerate(chars):
        if random.random() < intensity and c.lower() in glitch_map:
            chars[i] = glitch_map[c.lower()]
    return "".join(chars)


def build_document(doc_id: int, force_ambiguous=False, force_duplicate_of=None):
    if force_ambiguous:
        template = random.choice(AMBIGUOUS_TEMPLATES)
        content = template["content"].format(
            n=random.randint(1, 99), amount=random.randint(100, 9999),
            date=f"2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
        )
        return {
            "content": content,
            "true_department": template["true_department"],
            "true_doc_type": template["true_doc_type"],
            "true_language": "EN",
            "true_sensitivity": SENSITIVITY_MAP[template["true_department"]],
            "is_ambiguous": True,
            "is_duplicate_of": None,
        }

    dept, doc_type = random.choice(list(TEMPLATES.keys()))
    template = random.choice(TEMPLATES[(dept, doc_type)])
    lang = random.choice(LANGUAGES) if random.random() > 0.3 else "EN"  # bias toward EN, but keep FR/AR present
    content = template.format(
        dept=dept, n=random.randint(1, 99), amount=random.randint(100, 9999),
        date=f"2026-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
    )

    # Occasionally inject OCR-style noise (simulating scanned docs)
    if random.random() < 0.15:
        content = inject_ocr_noise(content)

    return {
        "content": content,
        "true_department": dept,
        "true_doc_type": doc_type,
        "true_language": lang,
        "true_sensitivity": SENSITIVITY_MAP[dept],
        "is_ambiguous": False,
        "is_duplicate_of": force_duplicate_of,
    }


def build_dataset(count: int, ambiguous_ratio=0.1, duplicate_ratio=0.1):
    docs = []
    n_ambiguous = max(1, int(count * ambiguous_ratio))
    n_duplicates = max(1, int(count * duplicate_ratio))
    n_regular = count - n_ambiguous - n_duplicates

    for i in range(n_regular):
        docs.append(build_document(i))

    for i in range(n_ambiguous):
        docs.append(build_document(n_regular + i, force_ambiguous=True))

    # Duplicates: pick an existing doc, either copy exactly or with a minor edit (near-dup)
    for i in range(n_duplicates):
        source = random.choice(docs)
        is_exact = random.random() < 0.5
        dup_content = source["content"] if is_exact else source["content"] + "\n\n[Revised copy — minor formatting update.]"
        docs.append({
            **source,
            "content": dup_content,
            "is_duplicate_of": docs.index(source),
        })

    random.shuffle(docs)
    return docs


EXT_BY_DOCTYPE = {"Invoice": "txt", "Report": "txt", "Policy": "txt", "Contract": "txt", "Email": "txt"}


def write_dataset(docs, out_dir: Path, labels_filename: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for i, doc in enumerate(docs):
        ext = EXT_BY_DOCTYPE.get(doc["true_doc_type"], "txt")
        filename = f"doc_{i:03d}_{doc['true_department'].lower()}_{doc['true_doc_type'].lower()}.{ext}"
        (out_dir / filename).write_text(doc["content"], encoding="utf-8")
        labels.append({"filename": filename, **{k: v for k, v in doc.items() if k != "content"}})

    with open(out_dir / labels_filename, "w", encoding="utf-8") as f:
        for l in labels:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    return labels


def print_distribution(labels, name):
    from collections import Counter
    print(f"\n--- {name} distribution ---")
    print("Department:", dict(Counter(l["true_department"] for l in labels)))
    print("Doc type:  ", dict(Counter(l["true_doc_type"] for l in labels)))
    print("Language:  ", dict(Counter(l["true_language"] for l in labels)))
    print("Ambiguous: ", sum(1 for l in labels if l.get("is_ambiguous")))
    print("Duplicates:", sum(1 for l in labels if l.get("is_duplicate_of") is not None))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev_count", type=int, default=30)
    parser.add_argument("--gold_count", type=int, default=60)
    parser.add_argument("--output_dir", default="./test_corpus")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.output_dir)

    dev_docs = build_dataset(args.dev_count)
    dev_labels = write_dataset(dev_docs, out_dir / "dev", "dev_labels.jsonl")
    print_distribution(dev_labels, "DEV SET (tune freely)")

    # Different seed offset for gold so it's not just a repeat of dev's random sequence
    random.seed(args.seed + 999)
    gold_docs = build_dataset(args.gold_count)
    gold_labels = write_dataset(gold_docs, out_dir / "gold", "gold_labels.jsonl")
    print_distribution(gold_labels, "GOLD SET (lock this — do not tune against it)")

    print(f"\nWritten to: {out_dir.resolve()}")
    print("Reminder: only run your classifier against gold/ ONCE, at the end, for your reported accuracy number.")
