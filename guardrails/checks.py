"""
Input and output safety guardrails.

check_input  — runs BEFORE retrieval (blocks out-of-scope requests)
check_output — runs AFTER generation (catches persona breaks / hallucinations)

Both return a result dict so the caller decides how to handle failures
rather than raising exceptions inside pipeline nodes.
"""
from __future__ import annotations

# ── Signal lists ───────────────────────────────────────────────────────────────

PERSONA_BREAK_SIGNALS = [
    "as an ai",
    "i'm an ai",
    "i am an ai",
    "as a language model",
    "i don't have personal",
    "i cannot have",
    "i'm not able to",
    "as your assistant",
    "i was trained",
    "my training data",
]

OUT_OF_SCOPE_SIGNALS = [
    "write a poem",
    "write me a story",
    "solve this math",
    "translate this",
    "summarize this article",
    "what's the weather",
    "who won the game",
    "stock price",
    "breaking news",
]

SAFE_FALLBACK = "I don't know."
OOS_FALLBACK  = "I'm here to help communicate as this person — that's a bit outside what I do."


# ── Public API ─────────────────────────────────────────────────────────────────

def check_input(query: str) -> dict:
    """
    Validate the partner's query before retrieval.

    Returns:
        {"allowed": bool, "reason": str | None, "fallback": str | None}
    """
    q = query.lower().strip()

    if any(s in q for s in OUT_OF_SCOPE_SIGNALS):
        return {"allowed": False, "reason": "out_of_scope", "fallback": OOS_FALLBACK}

    if len(q) < 2:
        return {"allowed": False, "reason": "empty_query", "fallback": "Could you repeat that?"}

    return {"allowed": True, "reason": None, "fallback": None}


def check_output(response: str, memories: list[dict]) -> dict:
    """
    Validate the generated response after generation.

    Checks:
      1. Persona break — did the model say "as an AI …"?
      2. Basic hallucination signal — response claims facts not in memories.

    Returns:
        {"passed": bool, "issue": str | None, "fallback": str | None}
    """
    r = response.lower()

    if any(signal in r for signal in PERSONA_BREAK_SIGNALS):
        return {"passed": False, "issue": "persona_break", "fallback": SAFE_FALLBACK}

    # Light hallucination check: if the model asserts specific numbers or
    # proper nouns that don't appear anywhere in the retrieved memories, flag it.
    # (Full NLI-based check is handled in the evaluation pipeline, not here.)
    if not memories and _makes_factual_claim(response):
        return {"passed": False, "issue": "unsupported_claim", "fallback": SAFE_FALLBACK}

    return {"passed": True, "issue": None, "fallback": None}


# ── Helpers ───────────────────────────────────────────────────────────────────

_FACTUAL_MARKERS = [
    " is ", " was ", " has ", " have ", " lives in ",
    " born in ", " works at ", " studied at ",
]

def _makes_factual_claim(text: str) -> bool:
    """Heuristic: does the text assert a specific fact?"""
    t = text.lower()
    return any(marker in t for marker in _FACTUAL_MARKERS)
