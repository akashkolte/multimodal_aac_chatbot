"""Shared intent-kind classification — used by retrieval, planner, and the
turnaround endpoint to decide whether a turn is memory-grounded or about
the user's present state.

Centralized here so all three callers stay in lock-step. If you add a new
intent type that should bypass retrieval, this is the only file to update.
"""


def classify_intent_kind(route: dict | None) -> str:
    """Return "present_state" if every sub-intent is PRESENT_STATE; "memory" otherwise.

    Mixed routes fall through to "memory" — the memory path is the safer default
    because it still allows the model to use chunks if any are present.
    """
    if not route:
        return "memory"
    sub_intents = route.get("sub_intents") or []
    if not sub_intents:
        return "memory"
    if all(si.get("type") == "PRESENT_STATE" for si in sub_intents):
        return "present_state"
    return "memory"


def is_present_state_only(route: dict | None) -> bool:
    """Convenience wrapper for retrieval node — returns True iff route is purely PRESENT_STATE."""
    return classify_intent_kind(route) == "present_state"
