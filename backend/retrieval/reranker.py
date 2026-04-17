import torch

from backend.pipeline.state import RetrievedChunk
from backend.retrieval.vector_store import get_device, get_embedder


def build_context_vector(
    raw_query: str,
    history: list[dict] | None,
    last_n_turns: int,
    weight_current: float,
) -> torch.Tensor:
    # Keep the current query dominant — silently dropping it (weight=0) makes
    # the reranker drift onto stale conversation topics.
    weight_current = max(0.05, min(1.0, weight_current))

    embedder = get_embedder()
    device = get_device()

    q_vec = embedder.encode(
        [raw_query],
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
    )[0]

    if not history or last_n_turns <= 0 or weight_current >= 1.0:
        return q_vec

    user_turns = [
        (h.get("content") or "").strip()
        for h in history
        if h.get("role") == "user" and (h.get("content") or "").strip()
    ]
    recent = user_turns[-last_n_turns:]
    if not recent:
        return q_vec

    h_vecs = embedder.encode(
        recent,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
    )
    h_mean = h_vecs.mean(dim=0)

    fused = weight_current * q_vec + (1.0 - weight_current) * h_mean
    return fused / fused.norm().clamp_min(1e-12)


def mmr_rerank(
    query_vec: torch.Tensor,
    candidate_vecs: torch.Tensor,
    candidate_chunks: list[RetrievedChunk],
    top_k: int,
    lambda_: float,
) -> list[RetrievedChunk]:
    n = candidate_vecs.shape[0]
    if n == 0 or top_k <= 0:
        return []
    if n <= top_k and lambda_ >= 1.0:
        return candidate_chunks[:top_k]

    rel = candidate_vecs @ query_vec  # (N,)
    pair = candidate_vecs @ candidate_vecs.T  # (N, N)
    target = min(top_k, n)

    NEG_INF = torch.full((), float("-inf"), device=rel.device, dtype=rel.dtype)
    available_mask = torch.ones(n, dtype=torch.bool, device=rel.device)
    max_sim_to_selected = torch.full(
        (n,), float("-inf"), device=rel.device, dtype=rel.dtype
    )

    selected: list[int] = []
    selected_scores: list[float] = []

    for step in range(target):
        if step == 0:
            scores = rel.clone()
        else:
            scores = lambda_ * rel - (1.0 - lambda_) * max_sim_to_selected
        masked = torch.where(available_mask, scores, NEG_INF)
        idx = int(torch.argmax(masked).item())
        selected.append(idx)
        selected_scores.append(float(scores[idx].item()))
        available_mask[idx] = False
        max_sim_to_selected = torch.maximum(max_sim_to_selected, pair[idx])

    out: list[RetrievedChunk] = []
    for idx, score in zip(selected, selected_scores):
        chunk = dict(candidate_chunks[idx])
        chunk["score"] = score
        out.append(chunk)  # type: ignore[arg-type]
    return out
