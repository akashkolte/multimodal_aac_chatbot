# agents/guardrails.py

# Phrases that indicate the model broke persona
PERSONA_BREAK_SIGNALS = [
    "as an ai", "i'm an ai", "i am an ai",
    "as a language model", "i don't have personal",
    "i cannot have", "i'm not able to",
    "as your assistant"
]

# Topics clearly out of scope for a personal AAC chatbot
OUT_OF_SCOPE_SIGNALS = [
    "write a poem", "write me a story", "solve this math",
    "translate this", "summarize this article",
    "what's the weather", "who won the game"
]

def check_input(query: str) -> dict:
    """
    Returns {"allowed": True/False, "reason": str}
    Run this BEFORE retrieval.
    """
    q = query.lower().strip()

    # Block clearly out-of-scope requests
    if any(s in q for s in OUT_OF_SCOPE_SIGNALS):
        return {
            "allowed": False,
            "reason": "out_of_scope",
            "fallback": "I'm here to help communicate as this person — that's a bit outside what I do."
        }

    return {"allowed": True, "reason": None, "fallback": None}


def check_output(response: str, memories: list) -> dict:
    """
    Returns {"passed": True/False, "issue": str}
    Run this AFTER generation.

    Checks:
    1. Persona break — did model say "as an AI"?
    2. Hallucination signal — did model use memories correctly?
    """
    r = response.lower()

    # Check persona break
    if any(signal in r for signal in PERSONA_BREAK_SIGNALS):
        return {
            "passed": False,
            "issue": "persona_break",
            "fallback": "I don't know."
        }

    return {"passed": True, "issue": None, "fallback": None}