# Session-level Bayesian bucket priors — updated after each accepted turn.
from __future__ import annotations

BUCKETS = ["family", "medical", "hobbies", "daily_routine", "social"]


def uniform_priors() -> dict[str, float]:
    p = 1.0 / len(BUCKETS)
    return {b: p for b in BUCKETS}


def update_priors(
    priors: dict[str, float],
    accepted_bucket: str,
    smoothing: float = 0.1,
) -> dict[str, float]:
    # Boost accepted bucket, normalise.
    if not priors:
        priors = uniform_priors()

    updated = {b: v + smoothing for b, v in priors.items()}
    updated[accepted_bucket] = updated.get(accepted_bucket, smoothing) + 1.0

    total = sum(updated.values())
    return {b: round(v / total, 6) for b, v in updated.items()}


def top_bucket(priors: dict[str, float]) -> str:
    if not priors:
        return BUCKETS[0]
    return max(priors, key=priors.get)
