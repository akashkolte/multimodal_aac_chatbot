# BGE embeddings + torch-tensor cosine search (mps → cuda → cpu).
import json
import math
from functools import lru_cache
from pathlib import Path

import torch

from backend.config.settings import settings
from backend.pipeline.state import RetrievedChunk
from backend.retrieval.priors import BUCKET_WEIGHT, TYPE_WEIGHT


def _prior_boost(
    chunk_meta: dict,
    bucket_priors: dict[str, float] | None,
    type_priors: dict[str, float] | None,
) -> float:
    b = 0.0
    if bucket_priors:
        b += BUCKET_WEIGHT * math.log(
            max(bucket_priors.get(chunk_meta["bucket"], 1e-3), 1e-3)
        )
    if type_priors:
        b += TYPE_WEIGHT * math.log(
            max(type_priors.get(chunk_meta.get("type", "narrative"), 1e-3), 1e-3)
        )
    return b


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


def get_embedder():
    return _get_embedder()


def embed_texts(texts: list[str]) -> torch.Tensor:
    return _get_embedder().encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=_DEVICE,
    )


# Index cache: one (vectors_tensor, meta) per user_id.
_index_cache: dict[str, tuple[torch.Tensor, list[dict]]] = {}


def load_index(user_id: str) -> tuple[torch.Tensor, list[dict]]:
    if user_id not in _index_cache:
        store_path = settings.vector_store_dir / user_id
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
    bucket_priors: dict[str, float] | None = None,
    type_priors: dict[str, float] | None = None,
    return_vectors: bool = False,
) -> list[RetrievedChunk] | tuple[list[RetrievedChunk], torch.Tensor]:
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

    # Priors rerank within this cosine top-k pool, not across all chunks —
    # top_k must be wide enough that favored labels have candidates here.
    candidates = [
        (top_scores_list[i], int(idx), meta[idx])
        for i, idx in enumerate(top_idxs_list)
        if 0 <= idx < len(meta)
    ]

    # Gaze is an explicit user signal — hard filter.
    if bucket_filter:
        filtered = [t for t in candidates if t[2]["bucket"] == bucket_filter]
        candidates = filtered if filtered else candidates  # fallback: all buckets

    # Soft-weight by log P(bucket) + log P(type); uniform priors are no-ops.
    if bucket_priors or type_priors:
        candidates = [
            (s + _prior_boost(c, bucket_priors, type_priors), idx, c)
            for s, idx, c in candidates
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)

    selected = candidates[:rerank_k]

    chunks = [
        RetrievedChunk(
            text=c["text"],
            bucket=c["bucket"],
            type=c.get("type", "narrative"),
            user=c["user"],
            score=float(s),
            source="personal",
        )
        for s, _, c in selected
    ]

    if return_vectors:
        if selected:
            sel_idxs = torch.tensor([idx for _, idx, _ in selected], device=_DEVICE)
            sel_vecs = vecs.index_select(0, sel_idxs)
        else:
            sel_vecs = torch.empty((0, vecs.shape[1]), device=_DEVICE)
        return chunks, sel_vecs

    return chunks


# Index builder.
def build_index(persona_path: str | Path) -> tuple[torch.Tensor, list[dict]]:
    with open(persona_path) as f:
        persona = json.load(f)

    user_name = persona["profile"]["name"]
    chunks, meta = [], []

    for bucket, memories in persona["memory_buckets"].items():
        for mem in memories:
            text = mem["text"]
            mem_type = mem.get("type", "narrative")
            chunks.append(text)
            meta.append(
                {"text": text, "bucket": bucket, "user": user_name, "type": mem_type}
            )

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
    store_dir = Path(store_dir or settings.vector_store_dir)

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
