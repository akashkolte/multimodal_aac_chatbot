"""Session-level Bayesian priors over a discrete label axis.

Used for both memory buckets (family / medical / ...) and chunk types
(narrative / social_post / chat_log). Each axis keeps its own P(label)
distribution, softly biases retrieval via log-weighted reranking, and
decays toward uniform so the prior tracks the current conversation,
not the whole session.
"""

DECAY = 0.15  # per-turn pull toward uniform (~5-turn half-life)
SMOOTHING = 0.1  # Laplace prior strength
BUCKET_WEIGHT = 0.3  # coefficient on log P(bucket) in retrieval rerank
TYPE_WEIGHT = 0.2  # coefficient on log P(type) in retrieval rerank

BUCKETS = ["family", "medical", "hobbies", "daily_routine", "social"]
CHUNK_TYPES = ["narrative", "social_post", "chat_log"]


def uniform(labels: list[str]) -> dict[str, float]:
    p = 1.0 / len(labels)
    return {b: p for b in labels}


def _decay(
    priors: dict[str, float], labels: list[str], decay: float
) -> dict[str, float]:
    u = 1.0 / len(labels)
    return {b: (1 - decay) * priors.get(b, u) + decay * u for b in labels}


def update_weighted(
    priors: dict[str, float],
    mass: dict[str, float],
    labels: list[str],
    smoothing: float = SMOOTHING,
    decay: float = DECAY,
) -> dict[str, float]:
    """Score-weighted Bayesian update with topic-drift decay.

    mass sums cosine scores per label across a turn's retrieved chunks.
    Strong matches move the prior more than weak ones; mixed turns update
    all contributing labels proportionally.
    """
    if not priors:
        priors = uniform(labels)

    decayed = _decay(priors, labels, decay)
    updated = {b: v + smoothing for b, v in decayed.items()}
    for label, m in mass.items():
        if m > 0 and label in updated:
            updated[label] += m

    total = sum(updated.values())
    return {b: round(v / total, 6) for b, v in updated.items()}
