# BGE embeddings + torch-tensor cosine search (mps → cuda → cpu).
import json
from functools import lru_cache
from pathlib import Path

import torch

from backend.config.settings import settings
from backend.pipeline.state import RetrievedChunk


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


_DEVICE = _select_device()


def get_device() -> str:
    return _DEVICE


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model, device=_DEVICE)


# Index cache: one (vectors_tensor, meta) per user_id.
_index_cache: dict[str, tuple[torch.Tensor, list[dict]]] = {}


def load_index(user_id: str) -> tuple[torch.Tensor, list[dict]]:
    if user_id not in _index_cache:
        store_path = settings.faiss_store_dir / user_id
        vecs = torch.load(
            store_path / "vectors.pt", map_location=_DEVICE, weights_only=True
        )
        with open(store_path / "meta.json") as f:
            meta = json.load(f)
        _index_cache[user_id] = (vecs, meta)
    return _index_cache[user_id]


# Retrieve.
def retrieve(
    query: str,
    user_id: str,
    top_k: int = 5,
    rerank_k: int = 3,
    bucket_filter: str | None = None,
) -> list[RetrievedChunk]:
    embedder = _get_embedder()
    vecs, meta = load_index(user_id)

    q_vec = embedder.encode(
        [query],
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=_DEVICE,
    )[0]

    scores = vecs @ q_vec  # cosine sim, vectors are L2-normalised
    k = min(top_k, scores.shape[0])
    top_scores, top_idxs = torch.topk(scores, k)
    top_scores_list = top_scores.tolist()
    top_idxs_list = top_idxs.tolist()

    candidates = [
        (top_scores_list[i], meta[idx])
        for i, idx in enumerate(top_idxs_list)
        if 0 <= idx < len(meta)
    ]

    if bucket_filter:
        filtered = [(s, c) for s, c in candidates if c["bucket"] == bucket_filter]
        candidates = filtered if filtered else candidates  # fallback: all buckets

    return [
        RetrievedChunk(
            text=c["text"], bucket=c["bucket"], user=c["user"], score=float(s)
        )
        for s, c in candidates[:rerank_k]
    ]


# Index builder.
def build_index(persona_path: str | Path) -> tuple[torch.Tensor, list[dict]]:
    with open(persona_path) as f:
        persona = json.load(f)

    user_name = persona["profile"]["name"]
    chunks, meta = [], []

    for bucket, memories in persona["memory_buckets"].items():
        for mem in memories:
            chunks.append(mem)
            meta.append({"text": mem, "bucket": bucket, "user": user_name})

    embedder = _get_embedder()
    vecs = embedder.encode(
        chunks,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=_DEVICE,
    )
    return vecs, meta


def save_index(vecs: torch.Tensor, meta: list[dict], save_dir: str | Path) -> None:
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    # Move to CPU before saving so the file is portable across devices.
    torch.save(vecs.detach().cpu(), p / "vectors.pt")
    with open(p / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def build_all(
    memories_dir: str | Path | None = None,
    store_dir: str | Path | None = None,
) -> None:
    memories_dir = Path(memories_dir or settings.memories_dir)
    store_dir = Path(store_dir or settings.faiss_store_dir)

    print(f"Embedder device: {_DEVICE}")
    for persona_file in sorted(memories_dir.glob("*.json")):
        uid = persona_file.stem
        print(f"  Building index for {uid} …")
        vecs, meta = build_index(persona_file)
        save_index(vecs, meta, store_dir / uid)
        print(f"    Saved {len(meta)} chunks → {store_dir / uid}/")
    print("\nAll indexes built.")


if __name__ == "__main__":
    build_all()
