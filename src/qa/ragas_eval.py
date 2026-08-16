"""
Lightweight RAGAS-style faithfulness judging.

This is NOT the full RAGAS library — it's a single, honest LLM-as-judge
metric: given a question, a generated answer, and the context that was
actually retrieved, ask a model "is this answer supported by this context?"
and get back a 0-1 score.

Design choices, deliberately mirroring the rest of the pipeline:
- No silent fallback. If no LLM is reachable, or the judge's response can't
  be parsed, that is reported as an explicit status ("unavailable" /
  "unparseable" / "error") — never averaged in as if it were a real score.
- Reuses the same provider/model candidate order as rag_pipeline.py, so the
  judge fails over the same way answer generation does (env override ->
  Gemini -> OpenAI -> local Ollama).
"""
import json
import os
import re
from typing import Any, Dict, List, Optional


def _get_judge_model_candidates() -> List[str]:
    raw_value = os.getenv("LITELLM_JUDGE_MODEL_NAMES") or os.getenv("LITELLM_MODEL_NAMES") or os.getenv("LITELLM_MODEL_NAME")
    explicit_candidates: List[str] = []
    if raw_value:
        explicit_candidates = [item.strip() for item in raw_value.split(",") if item.strip()]
    if explicit_candidates:
        return explicit_candidates
    if os.getenv("GEMINI_API_KEY"):
        return ["gemini/gemini-3.5-flash-lite"]
    if os.getenv("OPENAI_API_KEY"):
        return ["gpt-4o-mini"]
    return ["ollama/qwen3.6"]


_JUDGE_PROMPT_TEMPLATE = """You are a strict faithfulness judge for a RAG system.

You will be given a QUESTION, an ANSWER produced by an AI assistant, and the \
CONTEXT that was retrieved and given to that assistant to answer from.

Your job: judge whether every factual claim in the ANSWER is actually \
supported by the CONTEXT. An answer that:
- states facts not present in the context (invented/hallucinated) is UNFAITHFUL
- correctly says the context doesn't contain the answer is FAITHFUL
- draws only on what's in the context is FAITHFUL

QUESTION: {question}

CONTEXT:
{context}

ANSWER: {answer}

Respond with ONLY a JSON object, no other text, in exactly this shape:
{{"score": <float between 0.0 and 1.0, where 1.0 is fully faithful and 0.0 is fully invented>, "reason": "<one short sentence>"}}"""


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object out of a model response, tolerating markdown
    code fences or minor surrounding text — but never guessing at a score
    if there's genuinely no valid JSON object present."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def judge_faithfulness(question: str, answer: str, context: str) -> Dict[str, Any]:
    """Judge whether `answer` is faithful to `context` given `question`.

    Returns a dict always containing "status", one of:
      - "judged"       -> "score" (0.0-1.0 float) and "reason" are present
      - "unavailable"  -> litellm isn't installed, no judge could run
      - "no_context"   -> context was empty, nothing to judge faithfulness against
      - "unparseable"  -> the judge model responded but not in valid JSON
      - "error"        -> every candidate model raised (network/auth/etc.)
    Never returns a fabricated score for the non-"judged" statuses, so
    callers (like the hallucination test) can tell "judged as faithful"
    apart from "couldn't judge at all".
    """
    if not context or not context.strip():
        return {"status": "no_context", "score": None, "reason": "no retrieved context to check the answer against"}

    if not answer or not answer.strip():
        return {"status": "judged", "score": 1.0, "reason": "empty answer makes no unsupported claims"}

    try:
        from litellm import completion
    except ImportError:
        return {"status": "unavailable", "score": None, "reason": "litellm not installed; judge could not run"}

    prompt = _JUDGE_PROMPT_TEMPLATE.format(question=question, context=context, answer=answer)
    completion_kwargs = {"temperature": 0.0, "timeout": 180}

    last_error: Optional[str] = None
    for model_name in _get_judge_model_candidates():
        call_kwargs = dict(completion_kwargs)
        if model_name.startswith("ollama/"):
            call_kwargs["api_base"] = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        try:
            response = completion(model=model_name, messages=[{"role": "user", "content": prompt}], **call_kwargs)
            raw_text = response.choices[0].message.content or ""
            parsed = _extract_json_object(raw_text)
            if parsed is None or "score" not in parsed:
                return {
                    "status": "unparseable",
                    "score": None,
                    "reason": f"judge model '{model_name}' did not return valid JSON with a score",
                    "raw_response": raw_text[:500],
                }
            try:
                score = float(parsed["score"])
            except (TypeError, ValueError):
                return {
                    "status": "unparseable",
                    "score": None,
                    "reason": f"judge model '{model_name}' returned a non-numeric score",
                    "raw_response": raw_text[:500],
                }
            score = max(0.0, min(1.0, score))
            return {
                "status": "judged",
                "score": score,
                "reason": parsed.get("reason", ""),
                "model_used": model_name,
            }
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we try the next candidate
            last_error = f"{type(exc).__name__}: {exc}"
            continue

    return {"status": "error", "score": None, "reason": last_error or "all judge model candidates failed"}
