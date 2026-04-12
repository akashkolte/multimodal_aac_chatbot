"""
Session-level Bayesian bucket priors (proposal §5.4 Bonus).

Prior P(bucket_i) is initialized uniformly across the 5 buckets.
After each accepted response, the prior is updated proportionally
to the historical acceptance rate for that bucket in the session.

P(bucket_i | accept) ∝ P(accept | bucket_i) · P(bucket_i)

The updated priors are stored in PipelineState and passed to the
retrieval node to bias FAISS search toward the most contextually
likely topic for the session.
"""
from __future__ import annotations

BUCKETS = ["family", "medical", "hobbies", "daily_routine", "social"]


def uniform_priors() -> dict[str, float]:
    """Return equal probability mass over all buckets."""
    p = 1.0 / len(BUCKETS)
    return {b: p for b in BUCKETS}


def update_priors(
    priors: dict[str, float],
    accepted_bucket: str,
    smoothing: float = 0.1,
) -> dict[str, float]:
    """
    Bayesian update: boost the accepted bucket, normalise.

    Args:
        priors:          Current session priors (must sum to ~1.0).
        accepted_bucket: Bucket that sourced the accepted response.
        smoothing:       Additive smoothing constant to prevent zero probabilities.
    """
    if not priors:
        priors = uniform_priors()

    updated = {b: v + smoothing for b, v in priors.items()}
    updated[accepted_bucket] = updated.get(accepted_bucket, smoothing) + 1.0

    total = sum(updated.values())
    return {b: round(v / total, 6) for b, v in updated.items()}


def top_bucket(priors: dict[str, float]) -> str:
    """Return the bucket with the highest prior."""
    if not priors:
        return BUCKETS[0]
    return max(priors, key=priors.get)
