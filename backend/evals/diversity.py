import torch


def compute_candidate_diversity(candidates: list[dict]) -> dict:
    """Mean pairwise cosine *distance* among candidate texts.

    1.0 = maximally different, 0.0 = identical paraphrases. Empty candidate
    texts are filtered out before encoding, so `n_candidates` in the result
    is the count of *non-empty* texts (may be < len(candidates)).
    """
    texts = [c.get("text", "").strip() for c in candidates]
    texts = [t for t in texts if t]
    n = len(texts)
    if n < 2:
        return {"candidate_diversity": 0.0, "n_candidates": n}

    from backend.retrieval.vector_store import embed_texts

    vecs = embed_texts(texts)
    sims = vecs @ vecs.T
    iu = torch.triu_indices(n, n, offset=1)
    return {
        "candidate_diversity": round(float(1.0 - sims[iu[0], iu[1]].mean().item()), 4),
        "n_candidates": n,
    }
