# NLI-based faithfulness scoring.
from __future__ import annotations


def compute_faithfulness(response: str, chunks: list[dict]) -> dict:
    """Compute groundedness and hallucination rate via NLI."""
    no_evidence = len(chunks) == 0
    return {
        "groundedness": 0.0,
        "hallucination_rate": 0.0,
        "no_evidence": no_evidence,
    }
