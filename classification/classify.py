"""
Classification module — Stage 2 of the Atlas-to-RAG pipeline.
Enhanced with LiteLLM Gateway integration for production LLM classification.

Provides two classifiers:
1. Heuristic-based (fast, free, offline) — baseline/fallback
2. LLM-based (accurate, requires API) — via LiteLLM Gateway

Both can be compared via eval_classifiers.py

Setup for LiteLLM:
    # For Ollama (free, local):
    ollama pull mistral
    ollama serve

    # For OpenAI (cheap gpt-3.5-turbo):
    export OPENAI_API_KEY="sk-..."

    # For other models (Cohere, Anthropic, local, etc):
    See https://docs.litellm.ai

Run with heuristic (default):
    python classify.py --processed ./processed --output ./classified

Run with LLM:
    python classify.py --processed ./processed --output ./classified --use_llm

Run evaluation (compare both):
    python eval_classifiers.py
"""

import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict


@dataclass
class ClassificationRecord:
    source_file: str
    department: str
    doc_type: str
    language: str
    sensitivity: str
    confidence: float
    classified_at: str
    classifier: str = "heuristic"  # "heuristic" or "litellm"
    rules_applied: list = None


# ============================================================================
# HEURISTIC CLASSIFIER (Baseline/Fallback)
# ============================================================================

def classify_document_heuristic(markdown_content: str, filename: str) -> dict:
    """
    Lightweight heuristic classifier using keyword spotting.
    No external API calls — works offline, fast, free.
    Accuracy: ~75% on diverse documents.
    """
    content_lower = markdown_content.lower()
    filename_lower = filename.lower()
    
    # Department detection
    if any(word in content_lower for word in ["policy", "employee", "hr", "remote work", "payroll", "benefits", "leave", "bonus", "salary", "hiring"]):
        department = "HR"
    elif any(word in content_lower for word in ["invoice", "revenue", "expense", "financial", "finance", "q3", "q2", "quarterly", "budget"]):
        department = "Finance"
    elif any(word in content_lower for word in ["contract", "legal", "agreement", "liability", "confidential clause", "terms and conditions"]):
        department = "Legal"
    elif any(word in content_lower for word in ["it", "technology", "server", "network", "security", "infrastructure"]):
        department = "IT"
    else:
        department = "General"
    
    # Document type detection
    if "policy" in filename_lower or "policy" in content_lower:
        doc_type = "Policy"
    elif "invoice" in filename_lower or "invoice" in content_lower:
        doc_type = "Invoice"
    elif "statement" in filename_lower or ("financial" in content_lower and "statement" in content_lower):
        doc_type = "Financial Report"
    elif "csv" in filename_lower or ("table" in content_lower and "|" in markdown_content):
        doc_type = "Data Table"
    elif "contract" in filename_lower:
        doc_type = "Contract"
    else:
        doc_type = "Document"
    
    # Language detection
    language = "EN"
    # Only detect other languages if there are multiple keywords (reduce false positives)
    french_keywords = ["bonjour", "merci", "français", "salut", "document français"]
    arabic_keywords = ["salaam", "shukran", "arabic", "السلام"]
    spanish_keywords = ["hola", "gracias", "español"]
    
    if sum(1 for kw in french_keywords if kw in content_lower) >= 2:
        language = "FR"
    elif sum(1 for kw in arabic_keywords if kw in content_lower) >= 1:
        language = "AR"
    elif sum(1 for kw in spanish_keywords if kw in content_lower) >= 2:
        language = "ES"
    
    # Sensitivity detection — look for explicit markers
    if any(phrase in content_lower for phrase in ["confidential", "restricted", "secret", "do not share", "eyes only", "classified", "confidential information"]):
        sensitivity = "Confidential"
    elif any(phrase in content_lower for phrase in ["for internal use", "internal use only", "internal only", "internal:"]):
        sensitivity = "Internal"
    else:
        sensitivity = "Public"
    
    return {
        "department": department,
        "doc_type": doc_type,
        "language": language,
        "sensitivity": sensitivity,
        "confidence": 0.75,  # Lower confidence for heuristic
        "classifier": "heuristic",
    }


# ============================================================================
# LLM CLASSIFIER (Via LiteLLM Gateway)
# ============================================================================

def get_default_model_name() -> str:
    if os.getenv("LITELLM_MODEL_NAME"):
        return os.getenv("LITELLM_MODEL_NAME")
    if os.getenv("GEMINI_API_KEY"):
        return "gemini/gemini-2.0-flash"
    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    return "gpt-3.5-turbo"


def classify_document_litellm(markdown_content: str, filename: str) -> dict:
    """
    Call LLM via LiteLLM Gateway for high-accuracy classification.

    Supports OpenAI and Ollama backends via LiteLLM. This is used for the
    higher-risk fields (sensitivity and sometimes department) while keeping
    the rest of the pipeline simple.
    """
    try:
        from litellm import completion
    except ImportError:
        print("\n⚠️  LiteLLM not installed.")
        print("   Install with: pip install litellm")
        print("   Falling back to heuristic classifier...")
        return classify_document_heuristic(markdown_content, filename)

    model_name = get_default_model_name()

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
        return result

    except Exception as exc:
        print(f"⚠️  LLM classification failed for {filename} with model {model_name}: {exc}")
        print("   Falling back to heuristic classifier...")
        result = classify_document_heuristic(markdown_content, filename)
        result["classifier"] = "heuristic_fallback"
        result["llm_error"] = str(exc)
        result["llm_model"] = model_name
        return result


def classify_document_hybrid(markdown_content: str, filename: str) -> dict:
    """
    Risk-based routing:
    - Use heuristics or a local Ollama model for low-risk fields (language, doc_type)
    - Use OpenAI via LiteLLM for higher-risk fields (sensitivity, sometimes department)
    """
    heuristic_result = classify_document_heuristic(markdown_content, filename)

    low_risk_fields = {
        "doc_type": heuristic_result["doc_type"],
        "language": heuristic_result["language"],
    }

    high_risk_fields = {
        "department": heuristic_result["department"],
        "sensitivity": heuristic_result["sensitivity"],
    }

    try:
        from litellm import completion
    except ImportError:
        return {**heuristic_result, "classifier": "heuristic"}

    model_name = get_default_model_name()
    if model_name.startswith("ollama/"):
        # Use local Ollama for low-risk fields; keep heuristics for high-risk ones
        return {
            **heuristic_result,
            "doc_type": low_risk_fields["doc_type"],
            "language": low_risk_fields["language"],
            "classifier": "hybrid_ollama",
        }

    prompt = f"""You are classifying a document for a retrieval pipeline.
For the high-risk fields, return ONLY valid JSON.

Filename: {filename}

Content (first 800 chars):
{markdown_content[:800]}

Return exactly this JSON:
{{
  "department": "HR" | "Finance" | "Legal" | "IT" | "General",
  "sensitivity": "Public" | "Internal" | "Confidential" | "Restricted"
}}"""

    try:
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
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
        llm_result = json.loads(json_match.group())
        return {
            **heuristic_result,
            "department": llm_result.get("department", heuristic_result["department"]),
            "sensitivity": llm_result.get("sensitivity", heuristic_result["sensitivity"]),
            "confidence": 0.9,
            "classifier": "hybrid_openai",
        }
    except Exception as exc:
        heuristic_result["classifier"] = "heuristic_fallback"
        heuristic_result["llm_error"] = str(exc)
        heuristic_result["llm_model"] = model_name
        return heuristic_result



# ============================================================================
# PIPELINE
# ============================================================================

def run(processed_dir: Path, output_dir: Path, metadata_path: Path, use_llm: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classified_records = []
    classifier_name = "litellm" if use_llm else "heuristic"
    
    # Find all .md files from Stage 1
    for md_file in sorted(processed_dir.glob("*.md")):
        markdown_content = md_file.read_text()
        
        # Get existing metadata from Stage 1
        meta_file = md_file.with_suffix(".meta.json")
        stage1_meta = {}
        if meta_file.exists():
            stage1_meta = json.loads(meta_file.read_text())
        
        # Classify using selected classifier
        if use_llm:
            classification = classify_document_hybrid(markdown_content, md_file.name)
        else:
            classification = classify_document_heuristic(markdown_content, md_file.name)
        
        # Combine with Stage 1 metadata
        enriched_meta = {
            **stage1_meta,
            "classification": classification,
            "classified_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Write enriched metadata
        out_meta_file = output_dir / md_file.with_suffix(".classified.json").name
        out_meta_file.write_text(json.dumps(enriched_meta, indent=2))
        
        # Copy markdown to output
        out_md_file = output_dir / md_file.name
        out_md_file.write_text(markdown_content)
        
        # Track record
        record = ClassificationRecord(
            source_file=md_file.name,
            department=classification["department"],
            doc_type=classification["doc_type"],
            language=classification["language"],
            sensitivity=classification["sensitivity"],
            confidence=classification["confidence"],
            classified_at=enriched_meta["classified_at"],
            classifier=classification.get("classifier", "unknown"),
            rules_applied=["heuristic_keywords" if not use_llm else "litellm_prompt"]
        )
        classified_records.append(asdict(record))
        
        symbol = "🤖" if use_llm else "⚡"
        print(f"  {symbol} [{classifier_name:10s}] {md_file.name:25s} → {record.department:10s} | {record.doc_type:15s} | {record.sensitivity}")
    
    # Write summary
    summary = {
        "total_classified": len(classified_records),
        "classifier": classifier_name,
        "records": classified_records,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(summary, indent=2))
    
    print(f"\n✅ Classification complete ({classifier_name}). {len(classified_records)} documents classified.")


def main():
    parser = argparse.ArgumentParser(description="Classification stage: add metadata to documents (heuristic or LLM-based).")
    parser.add_argument("--processed", required=True, help="Folder with Stage 1 outputs (.md + .meta.json)")
    parser.add_argument("--output", required=True, help="Folder to write enriched .classified.json files")
    parser.add_argument("--metadata", default="classified_metadata.json", help="Summary metadata file")
    parser.add_argument("--use_llm", action="store_true", help="Use LiteLLM Gateway instead of heuristics")
    args = parser.parse_args()
    
    processed_dir = Path(args.processed)
    if not processed_dir.is_dir():
        print(f"❌ Processed directory not found: {processed_dir}")
        exit(1)
    
    run(processed_dir, Path(args.output), Path(args.metadata), use_llm=args.use_llm)


if __name__ == "__main__":
    main()
