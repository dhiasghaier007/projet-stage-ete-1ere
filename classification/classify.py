"""
Classification module — Stage 2 of the Atlas-to-RAG pipeline.
Uses a strict LLM-only classifier via LiteLLM Gateway.

No heuristic fallback is used for actual classification. If the LLM fails
(auth error, access denied, rate limit, etc.), the document is marked as
failed and surfaced explicitly.
"""

import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ClassificationRecord:
    source_file: str
    department: str | None = None
    doc_type: str | None = None
    language: str | None = None
    sensitivity: str | None = None
    confidence: float = 0.0
    classified_at: str | None = None
    classifier: str = "litellm"
    rules_applied: list | None = None
    status: str = "classified"
    error: str | None = None


# ============================================================================
# LLM-ONLY CLASSIFIER (Via LiteLLM Gateway)
# ============================================================================

def get_model_candidates() -> list[str]:
    raw_value = os.getenv("LITELLM_MODEL_NAMES") or os.getenv("LITELLM_MODEL_NAME")
    if raw_value:
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    if os.getenv("GEMINI_API_KEY"):
        return ["gemini/gemini-2.0-flash"]
    if os.getenv("OPENAI_API_KEY"):
        return ["gpt-4o-mini"]
    return ["gpt-3.5-turbo"]


def get_default_model_name() -> str:
    return get_model_candidates()[0]


def classify_document_litellm(markdown_content: str, filename: str) -> dict:
    """Classify a document using the LLM only. On failure, return an explicit failure record."""
    try:
        from litellm import completion
    except ImportError as exc:
        print("\n🚨 LLM classifier unavailable: LiteLLM is not installed.")
        return {
            "status": "failed",
            "classifier": "litellm",
            "confidence": 0.0,
            "error": f"LiteLLM import failed: {exc}",
            "llm_model": None,
        }

    prompt = f"""You are classifying a document for a retrieval pipeline.
Return ONLY valid JSON with no markdown fences and no extra text.

Filename: {filename}

Content (first 800 chars):
{markdown_content[:800]}

Return exactly this JSON object:
{{
  "department": "HR" | "Finance" | "Legal" | "IT" | "General",
  "doc_type": "Policy" | "Invoice" | "Report" | "Contract" | "Data Table" | "Document",
  "language": "EN" | "FR" | "AR" | "ES",
  "sensitivity": "Public" | "Internal" | "Confidential" | "Restricted"
}}"""

    last_error = None
    for model_name in get_model_candidates():
        try:
            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                timeout=15,
            )

            response_text = response.choices[0].message.content.strip()
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON in response")

            result = json.loads(json_match.group())
            result["confidence"] = 0.92
            result["classifier"] = "litellm"
            result["llm_model"] = model_name
            result["status"] = "classified"
            return result

        except Exception as exc:
            last_error = exc
            print(f"🚨 LLM classification failed for {filename} with model {model_name}: {exc}")

    print("🚨 All configured LLM models failed; marking document as failed.")
    return {
        "status": "failed",
        "classifier": "litellm",
        "confidence": 0.0,
        "error": str(last_error),
        "llm_model": get_model_candidates()[-1],
    }


# ============================================================================
# PIPELINE
# ============================================================================

def run(processed_dir: Path, output_dir: Path, metadata_path: Path, use_llm: bool = True) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    classified_records = []
    classifier_name = "litellm"

    for md_file in sorted(processed_dir.glob("*.md")):
        markdown_content = md_file.read_text(encoding="utf-8", errors="ignore")

        meta_file = md_file.with_suffix(".meta.json")
        stage1_meta = {}
        if meta_file.exists():
            stage1_meta = json.loads(meta_file.read_text(encoding="utf-8"))

        classification = classify_document_litellm(markdown_content, md_file.name)

        enriched_meta = {
            **stage1_meta,
            "classification": classification,
            "classified_at": datetime.now(timezone.utc).isoformat(),
        }

        out_meta_file = output_dir / md_file.with_suffix(".classified.json").name
        out_meta_file.write_text(json.dumps(enriched_meta, indent=2), encoding="utf-8")

        out_md_file = output_dir / md_file.name
        out_md_file.write_text(markdown_content, encoding="utf-8")

        record = ClassificationRecord(
            source_file=md_file.name,
            department=classification.get("department"),
            doc_type=classification.get("doc_type"),
            language=classification.get("language"),
            sensitivity=classification.get("sensitivity"),
            confidence=classification.get("confidence", 0.0),
            classified_at=enriched_meta["classified_at"],
            classifier=classification.get("classifier", "litellm"),
            rules_applied=["litellm_prompt"],
            status=classification.get("status", "classified"),
            error=classification.get("error"),
        )
        classified_records.append(asdict(record))

        print(f"  🤖 [{classifier_name:10s}] {md_file.name:25s} → {classification.get('status', 'classified')}")

    summary = {
        "total_classified": len(classified_records),
        "classifier": classifier_name,
        "records": classified_records,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n✅ Classification complete ({classifier_name}). {len(classified_records)} documents classified.")


def main():
    parser = argparse.ArgumentParser(description="Classification stage: add metadata to documents using the LLM only.")
    parser.add_argument("--processed", required=True, help="Folder with Stage 1 outputs (.md + .meta.json)")
    parser.add_argument("--output", required=True, help="Folder to write enriched .classified.json files")
    parser.add_argument("--metadata", default="classified_metadata.json", help="Summary metadata file")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    if not processed_dir.is_dir():
        print(f"❌ Processed directory not found: {processed_dir}")
        exit(1)

    run(processed_dir, Path(args.output), Path(args.metadata))


if __name__ == "__main__":
    main()
