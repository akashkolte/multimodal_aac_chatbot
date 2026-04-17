import torch

from backend.pipeline.state import RetrievedChunk
from backend.retrieval.vector_store import get_device, get_embedder


def retrieve_from_history(
    query: str,
    history: list[dict],
    top_k: int = 3,
    recent_window: int = 20,
) -> list[RetrievedChunk]:
    if not history or top_k <= 0:
        return []

    window = history[-recent_window:]
    texts = [_format_turn(h) for h in window]
    if not any(texts):
        return []

    embedder = get_embedder()
    device = get_device()

    q_vec = embedder.encode(
        [query],
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
    )[0]
    h_vecs = embedder.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
    )

    scores = h_vecs @ q_vec
    k = min(top_k, scores.shape[0])
    top_scores, top_idxs = torch.topk(scores, k)

    return [
        RetrievedChunk(
            text=texts[int(idx)],
            bucket="contextual",
            type="chat_log",
            user="",
            score=float(score),
            source="contextual",
        )
        for score, idx in zip(top_scores.tolist(), top_idxs.tolist())
    ]


def _format_turn(turn: dict) -> str:
    role = turn.get("role", "?")
    content = (turn.get("content") or "").strip()
    return f"{role}: {content}" if content else ""
