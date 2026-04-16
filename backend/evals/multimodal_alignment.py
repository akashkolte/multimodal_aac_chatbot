# Multimodal alignment scoring.
from __future__ import annotations


def compute_multimodal_alignment(
    response: str,
    affect: str | None,
    gesture_tag: str | None,
    gaze_bucket: str | None,
    chunks: list[dict],
) -> dict:
    """Score alignment between non-verbal inputs and generated text."""
    return {
        "overall_score": 0.0,
        "affect_alignment": 0.0,
        "gesture_alignment": 0.0,
        "gaze_alignment": 0.0,
    }
