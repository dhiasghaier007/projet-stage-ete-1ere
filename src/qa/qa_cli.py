import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.qa.rag_pipeline import answer_question, evaluate_rag
from src.retrieval.access_control import DEFAULT_CLEARANCE, SENSITIVITY_LEVELS, max_sensitivity_of_chunks, ALL_DEPARTMENTS


MAX_HISTORY_TURNS = 5

# How much of a chunk's content to preview in verbose retrieval output —
# enough to sanity-check relevance without dumping the whole chunk into
# the terminal.
_CHUNK_PREVIEW_CHARS = 100


def _print_retrieval_details(result: dict) -> None:
    """Print what actually got searched and retrieved this turn: the
    (possibly rewritten) query, retrieval mode, and each chunk's score,
    sensitivity label, and source — so a person can verify *why* an answer
    came out the way it did, instead of just trusting the final text.
    This was previously only visible in --question mode's full JSON dump;
    interactive mode gave no way to see it without leaving the session.
    """
    print("  --- retrieval details ---")

    if result.get("query_rewrite_status") == "rewritten":
        print(f"  Rewritten query: \"{result['retrieval_query']}\" (from: \"{result['question']}\")")
    elif result.get("query_rewrite_status") not in (None, "no_history", "unnecessary"):
        # rewriting was attempted but failed (unavailable/unparseable/error) —
        # worth flagging, since it means retrieval used the raw follow-up
        # wording, which may have missed the right chunk on a vague question.
        print(f"  ⚠️  Query rewrite did not run (status: {result['query_rewrite_status']}); "
              f"searched with the question as typed.")

    print(f"  Retrieval mode: {result['retrieval_mode']}")

    chunks = result.get("retrieved_chunks", [])
    if not chunks:
        print("  No chunks retrieved.")
    else:
        print(f"  Retrieved {len(chunks)} chunk(s):")
        for idx, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source_file", "unknown source")
            sensitivity = metadata.get("sensitivity", "unlabeled")
            department = metadata.get("department", "unlabeled")
            score = chunk.get("rrf_score", 0.0)
            preview = (chunk.get("content", "") or "").strip().replace("\n", " ")
            if len(preview) > _CHUNK_PREVIEW_CHARS:
                preview = preview[:_CHUNK_PREVIEW_CHARS] + "..."
            print(f"    [{idx}] score={score:.4f}  sensitivity={sensitivity}  department={department}  source={source}")
            print(f"        \"{preview}\"")
    print("  --------------------------\n")


def _parse_departments(raw: str) -> Any:
    """Parse the --department CLI value or an interactive 'department ...'
    command into either ALL_DEPARTMENTS (unrestricted) or a list of
    department names. "All"/"all" (case-insensitive) is the explicit
    unrestricted sentinel; anything else is split on commas."""
    stripped = raw.strip()
    if stripped.lower() == "all":
        return ALL_DEPARTMENTS
    return [d.strip() for d in stripped.split(",") if d.strip()]


def _run_single_question(args, dsn: str) -> None:
    departments = _parse_departments(args.department)
    result = answer_question(args.index, args.question, top_k=args.top_k, pgvector_dsn=dsn, clearance=args.clearance, departments=departments, by_department=args.by_department)
    print(json.dumps({"result": result, "eval": evaluate_rag(result)}, indent=2))


def _run_interactive(args, dsn: str) -> None:
    verbose = args.verbose
    clearance = args.clearance
    departments = _parse_departments(args.department)
    departments_display = "All (unrestricted)" if departments == ALL_DEPARTMENTS else ", ".join(departments) or "None (no department, only shared/General content)"
    print("Interactive mode — type a question and press Enter. Type 'exit' or 'quit' to stop.")
    print(f"(Remembering the last {MAX_HISTORY_TURNS} turns of this session for follow-up context.)")
    print(f"(Clearance: {clearance} — you will not see chunks above this sensitivity level.)")
    print(f"(Department access: {departments_display} — plus shared/General content, always visible.)")
    print(f"(Retrieval details: {'ON' if verbose else 'off'} — type 'verbose on' / 'verbose off' to toggle.)")
    print(f"(Type 'clearance <{'/'.join(SENSITIVITY_LEVELS.keys())}>' to change clearance mid-session — "
          f"conversation history above your new level is automatically hidden, not just new retrieval.)")
    print("(Type 'department <Name1,Name2>' or 'department All' to change department access mid-session.)\n")
    conversation_history: list[dict] = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding session.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Ending session.")
            break
        if question.lower() == "reset":
            conversation_history = []
            print("(conversation history cleared)\n")
            continue
        if question.lower() in {"verbose on", "verbose"}:
            verbose = True
            print("(retrieval details: ON)\n")
            continue
        if question.lower() == "verbose off":
            verbose = False
            print("(retrieval details: off)\n")
            continue
        if question.lower().startswith("clearance "):
            requested = question.split(" ", 1)[1].strip()
            matched = next((lvl for lvl in SENSITIVITY_LEVELS if lvl.lower() == requested.lower()), None)
            if matched is None:
                print(f"(unrecognized clearance '{requested}' — choose one of: {', '.join(SENSITIVITY_LEVELS.keys())})\n")
            else:
                clearance = matched
                print(f"(clearance changed to {clearance}. Earlier answers above this level, if any, "
                      f"will no longer be usable as context for follow-ups.)\n")
            continue
        if question.lower().startswith("department "):
            requested = question.split(" ", 1)[1].strip()
            departments = _parse_departments(requested)
            departments_display = "All (unrestricted)" if departments == ALL_DEPARTMENTS else ", ".join(departments) or "None (only shared/General content)"
            print(f"(department access changed to: {departments_display})\n")
            continue

        result = answer_question(
            args.index,
            question,
            top_k=args.top_k,
            pgvector_dsn=dsn,
            conversation_history=conversation_history,
            clearance=clearance,
            departments=departments,
            by_department=args.by_department,
        )

        if verbose:
            print()
            _print_retrieval_details(result)

        print(f"Assistant: {result['answer']}\n")

        turn_sensitivity = max_sensitivity_of_chunks(result.get("retrieved_chunks", []))
        conversation_history.append({
            "question": question,
            "answer": result["answer"],
            "sensitivity_level": turn_sensitivity,
        })
        conversation_history = conversation_history[-MAX_HISTORY_TURNS:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval-augmented QA over a local vector index")
    parser.add_argument("--index", required=True, help="Path to the local_index.json file (used as fallback if Postgres is unreachable)")
    parser.add_argument("--question", default=None, help="Question to ask (single-shot mode; omit this to use --interactive instead)")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved chunks")
    parser.add_argument("--pgvector-dsn", default=None, help="Postgres DSN; defaults to PGVECTOR_DSN in .env if not set")
    parser.add_argument("--interactive", action="store_true", help="Start a multi-turn session that remembers previous questions/answers")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In --interactive mode, show retrieval details (rewritten query, retrieval mode, "
             "chunk scores/sources) after every answer. Can also be toggled live with 'verbose on'/'verbose off'.",
    )
    parser.add_argument(
        "--clearance",
        default=DEFAULT_CLEARANCE,
        choices=list(SENSITIVITY_LEVELS.keys()),
        help=f"Sensitivity clearance for this session (default: {DEFAULT_CLEARANCE}). "
             "Chunks labeled above this level are excluded from retrieval entirely.",
    )
    parser.add_argument(
        "--department",
        default="All",
        help="Comma-separated department access for this session, e.g. 'HR,Finance', or 'All' for "
             "unrestricted (default: All). Shared/General content is always visible regardless. "
             "Can also be changed mid-session in --interactive with 'department <Name1,Name2>'.",
    )
    parser.add_argument(
        "--by-department",
        action="store_true",
        help="Query real per-department storage (one pgvector table / local index per department) "
             "instead of one shared table. With this flag, --index must point at the directory "
             "produced by `index_cli.py --by-department` (containing department_index_manifest.json), "
             "not a single local_index.json file.",
    )
    args = parser.parse_args()

    if not args.interactive and not args.question:
        parser.error("provide --question for a single-shot run, or pass --interactive for a multi-turn session")

    dsn = args.pgvector_dsn or os.getenv("PGVECTOR_DSN") or os.getenv("POSTGRES_DSN")

    if args.interactive:
        _run_interactive(args, dsn)
    else:
        _run_single_question(args, dsn)


if __name__ == "__main__":
    main()
