"""
Sensitivity-based access control for retrieval.

Every chunk carries a `sensitivity` label in its metadata (Public, Internal,
Confidential, Restricted — the labels Stage 2 classification already
assigns). Until now nothing enforced it at retrieval time: a "Restricted"
document was exactly as retrievable as a "Public" one. This module is the
single source of truth for what a given clearance level is allowed to see,
used by both the local (in-memory) and Postgres retrieval paths so they
enforce identically.

Fail-closed by design: an unrecognized or missing sensitivity label is
treated as the most restrictive level, not the least. Silently letting an
unlabeled chunk through would be the exact kind of silent, hard-to-audit
failure this codebase otherwise deliberately avoids (see the ingestion/
classification "explicit failure state" pattern) — and for an access
control filter specifically, failing open is a security bug, not a
convenience.
"""
from typing import Any, Dict, List

# Ordinal ranking, lowest (most open) to highest (most restricted).
SENSITIVITY_LEVELS: Dict[str, int] = {
    "Public": 0,
    "Internal": 1,
    "Confidential": 2,
    "Restricted": 3,
}

# The rank an unrecognized/missing sensitivity label is treated as.
# Set to the highest level that exists so unknown data is never accidentally
# exposed just because it's unlabeled or mislabeled.
_UNKNOWN_SENSITIVITY_RANK = max(SENSITIVITY_LEVELS.values())

DEFAULT_CLEARANCE = "Internal"


def clearance_rank(clearance: str) -> int:
    """Rank for a clearance level. Unrecognized clearance names are treated
    as Public (rank 0) — fail-closed for the *caller's* clearance too, so a
    typo'd or unknown clearance can't accidentally grant broad access."""
    return SENSITIVITY_LEVELS.get(clearance, 0)


def is_allowed(sensitivity_label: Any, clearance: str) -> bool:
    """True if a chunk labeled `sensitivity_label` may be shown to someone
    with `clearance`. Missing/unrecognized labels are always denied except
    to the highest clearance level, matching the fail-closed policy above."""
    label_rank = SENSITIVITY_LEVELS.get(sensitivity_label, _UNKNOWN_SENSITIVITY_RANK)
    return label_rank <= clearance_rank(clearance)


def allowed_sensitivity_labels(clearance: str) -> List[str]:
    """The concrete list of sensitivity label strings a given clearance may
    see — used to build a SQL `sensitivity = ANY(%s)` filter. Deliberately
    does NOT include a catch-all for unrecognized labels (those are denied
    by is_allowed above and have no business appearing in this list)."""
    max_rank = clearance_rank(clearance)
    return [label for label, rank in SENSITIVITY_LEVELS.items() if rank <= max_rank]


def filter_chunks_by_clearance(chunks: List[Dict[str, Any]], clearance: str) -> List[Dict[str, Any]]:
    """Filter a list of retrieval-result dicts (each with a "metadata" key
    containing "sensitivity") down to what `clearance` is allowed to see.
    Used as the local-index enforcement point and as a defensive
    second check even on the Postgres path, in case the SQL filter and this
    module's rules ever drift apart."""
    return [
        chunk for chunk in chunks
        if is_allowed(chunk.get("metadata", {}).get("sensitivity"), clearance)
    ]


def max_sensitivity_of_chunks(chunks: List[Dict[str, Any]]) -> str:
    """The highest sensitivity label among a set of retrieved chunks — used
    to tag a conversation turn with "how sensitive was the information that
    actually informed this answer". Chunks passed in here have normally
    already been through filter_chunks_by_clearance, so their labels should
    already be recognized ones; if an unrecognized label somehow shows up
    anyway, it's treated as the most restrictive rank for THIS calculation
    too — better to over-tag a turn as more sensitive than it was than to
    under-tag it and let it leak into a later, lower-clearance turn's
    context. Returns "Public" (the least restrictive label) if no chunks
    were retrieved at all, since no sensitive information was used."""
    if not chunks:
        return "Public"
    max_rank = -1
    max_label = "Public"
    for chunk in chunks:
        label = chunk.get("metadata", {}).get("sensitivity")
        rank = SENSITIVITY_LEVELS.get(label, _UNKNOWN_SENSITIVITY_RANK)
        if rank > max_rank:
            max_rank = rank
            max_label = label if label in SENSITIVITY_LEVELS else "Restricted"
    return max_label


def filter_history_by_clearance(history: List[Dict[str, Any]], clearance: str) -> List[Dict[str, Any]]:
    """Filter conversation history turns down to what `clearance` may still
    see. Each turn is expected to carry a "sensitivity_level" key (set by
    the caller when the turn was recorded — see max_sensitivity_of_chunks)
    recording the highest sensitivity of chunks that informed that turn's
    answer.

    This exists so conversation memory can never become a side-channel
    around the access-control filter: even though the current CLI keeps
    clearance fixed for a whole session (so this can't be exploited today),
    memory content is otherwise completely independent of the retrieval
    filter that produced it. If clearance is ever made dynamic within a
    session — re-authentication, a future API layer, etc — a Confidential
    answer from turn 1 must not silently leak into a Public-clearance
    turn 5 just because it's sitting in conversation_history. Turns missing
    a "sensitivity_level" tag (e.g. old-format history) are fail-closed:
    excluded unless clearance is the highest level, same policy as
    is_allowed above.
    """
    visible = []
    for turn in history:
        level = turn.get("sensitivity_level")
        if level is None:
            # No recorded level — fail closed, same as an unlabeled chunk.
            if clearance_rank(clearance) >= _UNKNOWN_SENSITIVITY_RANK:
                visible.append(turn)
            continue
        if is_allowed(level, clearance):
            visible.append(turn)
    return visible
