# Input + output safety guardrails.
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
OOS_FALLBACK = (
    "I'm here to help communicate as this person — that's a bit outside what I do."
)


# ── Public API ─────────────────────────────────────────────────────────────────


def check_input(query: str) -> dict:
    q = query.lower().strip()

    if any(s in q for s in OUT_OF_SCOPE_SIGNALS):
        return {"allowed": False, "reason": "out_of_scope", "fallback": OOS_FALLBACK}

    if len(q) < 2:
        return {
            "allowed": False,
            "reason": "empty_query",
            "fallback": "Could you repeat that?",
        }

    return {"allowed": True, "reason": None, "fallback": None}


def check_output(response: str, memories: list[dict]) -> dict:
    r = response.lower()

    if any(signal in r for signal in PERSONA_BREAK_SIGNALS):
        return {"passed": False, "issue": "persona_break", "fallback": SAFE_FALLBACK}

    # Flag unsupported factual claims when no memories were retrieved.
    if not memories and _makes_factual_claim(response):
        return {
            "passed": False,
            "issue": "unsupported_claim",
            "fallback": SAFE_FALLBACK,
        }

    return {"passed": True, "issue": None, "fallback": None}


# ── Helpers ───────────────────────────────────────────────────────────────────

_FACTUAL_MARKERS = [
    " is ",
    " was ",
    " has ",
    " have ",
    " lives in ",
    " born in ",
    " works at ",
    " studied at ",
]


def _makes_factual_claim(text: str) -> bool:
    t = text.lower()
    return any(marker in t for marker in _FACTUAL_MARKERS)
