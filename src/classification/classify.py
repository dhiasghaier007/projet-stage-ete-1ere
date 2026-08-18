"""
Classification module — Stage 2 of the Atlas-to-RAG pipeline.
LLM-only classification via LiteLLM Gateway (OpenAI / Gemini / Ollama / DeepSeek etc).

The heuristic keyword classifier has been removed from the active pipeline.
There is NO silent fallback: if the LLM call fails (bad key, access denied,
rate limit, network error, etc.), that document is marked as UNCLASSIFIED
and a clear, loud notification is printed and stored — nothing is quietly
guessed at.

Setup:
    export LITELLM_MODEL_NAMES="gemini/gemini-3.5-flash-lite"   # or ollama/qwen3.6, gpt-4o-mini, etc.
    export GEMINI_API_KEY="..."      # matching whichever provider you're using
    export OLLAMA_API_BASE="http://<host>:11434"   # only needed for ollama/* models

Run:
    python classify.py --processed ./processed --output ./classified
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, asdict

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency may be absent
    def load_dotenv() -> bool:
        return False

from src.classification.rule_validator import validate_classification, validate_batch

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env", override=True)

# Free-tier Gemini models (e.g. gemini-3.5-flash-lite) are limited to ~15
# requests/minute. A small delay between calls keeps us under that limit
# instead of burning through the quota in the first few seconds.
REQUEST_DELAY_SECONDS = float(os.getenv("CLASSIFY_REQUEST_DELAY", "4.5"))


@dataclass
class ClassificationRecord:
    source_file: str
    department: str
    doc_type: str
    language: str
    sensitivity: str
    confidence: float
    classified_at: str
    classifier: str = "litellm"
    llm_model: str = ""
    error: str = ""


# ============================================================================
# LLM CLASSIFIER (Via LiteLLM Gateway) — the only classifier in the pipeline
# ============================================================================

def get_model_candidates() -> list[str]:
    raw_value = os.getenv("LITELLM_MODEL_NAMES") or os.getenv("LITELLM_MODEL_NAME")
    explicit_candidates: list[str] = []
    if raw_value:
        explicit_candidates = [item.strip() for item in raw_value.split(",") if item.strip()]

    candidates: list[str] = []

    if os.getenv("GEMINI_API_KEY"):
        candidates.append("gemini/gemini-3.5-flash-lite")
    if os.getenv("OPENAI_API_KEY"):
        candidates.append("gpt-4o-mini")

    if explicit_candidates:
        if any(model.startswith(("gemini/", "gpt", "openai/")) for model in explicit_candidates):
            return explicit_candidates
        candidates.extend(explicit_candidates)
        return candidates or ["ollama/qwen3.6"]

    if os.getenv("OLLAMA_API_BASE") or os.getenv("OLLAMA_HOST") or True:
        candidates.append("ollama/qwen3.6")

    return candidates or ["ollama/qwen3.6"]


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg


def _is_access_denied(exc: Exception) -> bool:
    """Detect auth/permission-style failures so we can flag them distinctly."""
    msg = str(exc).lower()
    signals = [
        "access denied", "unauthorized", "invalid api key", "invalid_api_key",
        "authentication", "permission", "403", "401", "api key not valid",
        "insufficient balance", "insufficient_quota",
    ]
    return any(s in msg for s in signals)


def classify_document_litellm(markdown_content: str, filename: str) -> dict:
    """
    Call LLM via LiteLLM Gateway for classification. If the LLM is unavailable
    or rejects credentials, fall back to a deterministic heuristic for local use.
    """
    try:
        from litellm import completion
    except ImportError:
        return {
            "department": "UNCLASSIFIED",
            "doc_type": "UNCLASSIFIED",
            "language": "UNCLASSIFIED",
            "sensitivity": "UNCLASSIFIED",
            "confidence": 0.0,
            "classifier": "llm_failed",
            "llm_model": "",
            "error": "LiteLLM is not installed",
            "access_denied": False,
        }

    prompt = f"""You are an expert document classifier for an enterprise retrieval pipeline.
Think briefly about the document, then return ONLY valid JSON with no markdown fences and no extra text after it.

CLASSIFICATION RULES:
1. Classify based on the PRIMARY SUBJECT MATTER of the document — what it is actually about.
   Do NOT classify based on keywords that merely appear inside a data table, example, or
   sample row (e.g. a table listing "HR" as one value among several departments is a
   generic data table, not an HR document itself).
2. For "language": base this STRICTLY on the actual language of the body text you are
   reading right now. Ignore any language name, tag, or label mentioned in the content
   or filename — judge only the real words in front of you.
3. For "sensitivity", use this scale precisely:
   - "Public": generic, non-sensitive content anyone could see (status reports, general emails).
   - "Internal": routine business content not meant for outside the company, but not
     legally or personally sensitive (internal memos, standard policies).
   - "Confidential": contains business-sensitive specifics that would cause harm if
     leaked (financial figures, strategic plans, personal employee data like salary).
   - "Restricted": contains legally binding, privileged, or highly regulated content
     (signed contracts, legal agreements, litigation material, regulatory filings) —
     use this for legal documents even if they don't use the word "confidential".
   Do not default to "Internal" just because a document is business-related, and do not
   default to "Confidential" for a legal contract when "Restricted" is the better fit.

EXAMPLES:

Example A (data table, not a topic document):
Content: "| dept | doc_type | language |\\n| HR | Policy | EN |\\n| Finance | Invoice | FR |"
Correct output: {{"department": "General", "doc_type": "Data Table", "language": "EN", "sensitivity": "Public"}}
(Reasoning: this is a generic sample/schema table, not an HR or Finance document — "HR" is a data value, not the topic.)

Example B (real policy document):
Content: "HR POLICY: Remote Work Guidelines. All employees may work remotely 3 days/week. For internal use only."
Correct output: {{"department": "HR", "doc_type": "Policy", "language": "EN", "sensitivity": "Internal"}}

Example C (plain public content, no sensitive markers):
Content: "System Uptime Report — Week 38. Overall uptime: 99.38%. 38 incidents recorded, resolved within SLA."
Correct output: {{"department": "IT", "doc_type": "Report", "language": "EN", "sensitivity": "Public"}}

Example D (legal contract — use Restricted, not just Confidential):
Content: "SERVICE AGREEMENT between Acme Corp and Beta LLC. This agreement and its terms are legally binding on both parties. Governing law: Delaware."
Correct output: {{"department": "Legal", "doc_type": "Contract", "language": "EN", "sensitivity": "Restricted"}}
(Reasoning: signed/binding legal agreements are Restricted, a stronger tier than Confidential.)

Now classify this real document:

Filename: {filename}

Content (first 1500 chars):
{markdown_content[:1500]}

Return exactly this JSON object, with a confidence 0.0-1.0 reflecting how sure you actually are:
{{
  "department": "HR" | "Finance" | "Legal" | "IT" | "General",
  "doc_type": "Policy" | "Invoice" | "Report" | "Contract" | "Data Table" | "Document" | "Email",
  "language": "EN" | "FR" | "AR" | "ES",
  "sensitivity": "Public" | "Internal" | "Confidential" | "Restricted",
  "confidence": 0.0
}}"""

    completion_kwargs = {}
    last_error = None
    last_model = None

    model_candidates = get_model_candidates()
    print(f"🔎 Attempting providers in order: {', '.join(model_candidates)}")

    for model_name in model_candidates:
        last_model = model_name
        if model_name.startswith("ollama/"):
            completion_kwargs = {"api_base": os.getenv("OLLAMA_API_BASE", "http://localhost:11434")}
            print(f"   → Trying Ollama backend '{model_name}' at {completion_kwargs['api_base']}")
        else:
            completion_kwargs = {}
            print(f"   → Trying provider '{model_name}'")

        try:
            response = completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=180,
                **completion_kwargs,
            )

            response_text = response.choices[0].message.content.strip()
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if not json_match:
                raise ValueError("No valid JSON in LLM response")

            result = json.loads(json_match.group())
            self_reported_confidence = result.get("confidence")
            try:
                confidence = float(self_reported_confidence)
                if not (0.0 <= confidence <= 1.0):
                    confidence = 0.85
            except (TypeError, ValueError):
                confidence = 0.85  # model didn't return a usable confidence

            return {
                "department": result.get("department", "UNCLASSIFIED"),
                "doc_type": result.get("doc_type", "UNCLASSIFIED"),
                "language": result.get("language", "UNCLASSIFIED"),
                "sensitivity": result.get("sensitivity", "UNCLASSIFIED"),
                "confidence": confidence,
                "classifier": "litellm",
                "llm_model": model_name,
                "error": "",
                "access_denied": False,
            }

        except Exception as exc:
            last_error = exc
            if _is_rate_limited(exc):
                print(f"⏳ RATE LIMITED for {filename} on '{model_name}' — waiting 20s and retrying once...")
                time.sleep(20)
                try:
                    response = completion(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1,
                        timeout=180,
                        **completion_kwargs,
                    )
                    response_text = response.choices[0].message.content.strip()
                    cleaned = response_text.strip()
                    if cleaned.startswith("```"):
                        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                        cleaned = re.sub(r'\s*```$', '', cleaned)
                    json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
                    if not json_match:
                        raise ValueError("No valid JSON in LLM response")
                    result = json.loads(json_match.group())
                    self_reported_confidence = result.get("confidence")
                    try:
                        confidence = float(self_reported_confidence)
                        if not (0.0 <= confidence <= 1.0):
                            confidence = 0.85
                    except (TypeError, ValueError):
                        confidence = 0.85
                    return {
                        "department": result.get("department", "UNCLASSIFIED"),
                        "doc_type": result.get("doc_type", "UNCLASSIFIED"),
                        "language": result.get("language", "UNCLASSIFIED"),
                        "sensitivity": result.get("sensitivity", "UNCLASSIFIED"),
                        "confidence": confidence,
                        "classifier": "litellm",
                        "llm_model": model_name,
                        "error": "",
                        "access_denied": False,
                    }
                except Exception as retry_exc:
                    last_error = retry_exc
                    print(f"❌ Retry also failed for {filename}: {retry_exc}")

            denied = _is_access_denied(exc)
            tag = "🔒 ACCESS DENIED" if denied else "⚠️  LLM CALL FAILED"
            print(f"{tag} for {filename} using model '{model_name}': {exc}")
            if len(model_candidates) > 1:
                print("   Trying next configured model...")

    denied = _is_access_denied(last_error) if last_error else False
    print(f"⚠️  LLM unavailable for {filename}; no classification produced.")
    return {
        "department": "UNCLASSIFIED",
        "doc_type": "UNCLASSIFIED",
        "language": "UNCLASSIFIED",
        "sensitivity": "UNCLASSIFIED",
        "confidence": 0.0,
        "classifier": "llm_failed",
        "llm_model": last_model or "",
        "error": str(last_error) if last_error else "LLM backend unavailable",
        "access_denied": denied,
    }


# ============================================================================
# PIPELINE
# ============================================================================

def run(processed_dir: Path, output_dir: Path, metadata_path: Path, limit: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    classified_records = []
    failed_count = 0
    access_denied_count = 0
    needs_review_count = 0

    md_files = sorted(processed_dir.glob("*.md"))
    if limit is not None:
        md_files = md_files[:limit]

    for md_file in md_files:
        markdown_content = md_file.read_text()

        meta_file = md_file.with_suffix(".meta.json")
        stage1_meta = {}
        if meta_file.exists():
            stage1_meta = json.loads(meta_file.read_text())

        classification = classify_document_litellm(markdown_content, md_file.name)

        if classification["classifier"] == "llm_failed":
            failed_count += 1
            if classification.get("access_denied"):
                access_denied_count += 1

        validation_result = validate_classification(classification)
        if validation_result.needs_review:
            needs_review_count += 1

        enriched_meta = {
            **stage1_meta,
            "classification": classification,
            "validation": validation_result.to_dict(),
            "classified_at": datetime.now(timezone.utc).isoformat(),
        }

        out_meta_file = output_dir / md_file.with_suffix(".classified.json").name
        out_meta_file.write_text(json.dumps(enriched_meta, indent=2))

        out_md_file = output_dir / md_file.name
        out_md_file.write_text(markdown_content)

        record = ClassificationRecord(
            source_file=md_file.name,
            department=classification["department"],
            doc_type=classification["doc_type"],
            language=classification["language"],
            sensitivity=classification["sensitivity"],
            confidence=classification["confidence"],
            classified_at=enriched_meta["classified_at"],
            classifier=classification["classifier"],
            llm_model=classification.get("llm_model", ""),
            error=classification.get("error", ""),
        )
        classified_records.append(asdict(record))

        symbol = "🤖" if classification["classifier"] == "litellm" else "❌"
        review_flag = "  🚩 NEEDS REVIEW" if validation_result.needs_review else ""
        print(f"  {symbol} [{classification['classifier']:12s}] {md_file.name:30s} → {record.department:12s} | {record.doc_type:15s} | {record.sensitivity}{review_flag}")

        time.sleep(REQUEST_DELAY_SECONDS)

    summary = {
        "total_classified": len(classified_records),
        "failed": failed_count,
        "access_denied": access_denied_count,
        "needs_review": needs_review_count,
        "classifier": "litellm",
        "records": classified_records,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(summary, indent=2))

    print(f"\n✅ Classification run complete. {len(classified_records)} documents processed, "
          f"{failed_count} failed ({access_denied_count} access-denied), "
          f"{needs_review_count} flagged for human review by rule-based validation.")
    if access_denied_count > 0:
        print("🔒 One or more documents failed due to ACCESS DENIED — check your API key / model config.")
    if needs_review_count > 0:
        print(f"🚩 {needs_review_count} document(s) failed a validation rule (see 'validation' field in "
              f"each .classified.json, or run rule_validator.py against {metadata_path} for a full report) "
              f"— review before these proceed to chunking/indexing.")


def main():
    parser = argparse.ArgumentParser(description="Classification stage: LLM-only classification via LiteLLM.")
    parser.add_argument("--processed", required=True, help="Folder with Stage 1 outputs (.md + .meta.json)")
    parser.add_argument("--output", required=True, help="Folder to write enriched .classified.json files")
    parser.add_argument("--metadata", default="classified_metadata.json", help="Summary metadata file")
    parser.add_argument("--limit", type=int, default=None, help="Optional quick smoke-test limit: process only the first N markdown files")
    args = parser.parse_args()

    processed_dir = Path(args.processed)
    if not processed_dir.is_dir():
        print(f"❌ Processed directory not found: {processed_dir}")
        exit(1)

    run(processed_dir, Path(args.output), Path(args.metadata), limit=args.limit)


if __name__ == "__main__":
    main()