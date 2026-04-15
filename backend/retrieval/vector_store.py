# FAISS retrieval with BGE embeddings and cross-encoder reranking.
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from backend.config.settings import settings
from backend.pipeline.state import RetrievedChunk


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embed_model)


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.rerank_model)


@lru_cache(maxsize=1)
def _get_faiss():
    import faiss

    return faiss


# ── Index cache (one FAISS index per user_id) ─────────────────────────────────

_index_cache: dict[str, tuple] = {}


def load_index(user_id: str):
    if user_id not in _index_cache:
        faiss = _get_faiss()
        store_path = settings.faiss_store_dir / user_id
        index = faiss.read_index(str(store_path / "index.faiss"))
        with open(store_path / "meta.json") as f:
            meta = json.load(f)
        _index_cache[user_id] = (index, meta)
    return _index_cache[user_id]


# ── Core retrieve function ─────────────────────────────────────────────────────


def retrieve(
    query: str,
    user_id: str,
    top_k: int = 5,
    rerank_k: int = 3,
    bucket_filter: str | None = None,
    use_reranker: bool = True,
) -> list[RetrievedChunk]:
    embedder = _get_embedder()
    index, meta = load_index(user_id)

    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    _, idxs = index.search(q_vec, top_k)

    candidates = [meta[i] for i in idxs[0] if 0 <= i < len(meta)]

    if bucket_filter:
        filtered = [c for c in candidates if c["bucket"] == bucket_filter]
        candidates = filtered if filtered else candidates  # fallback: all buckets

    if use_reranker and len(candidates) > 1:
        reranker = _get_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = reranker.predict(pairs)
        ranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
        top = [
            RetrievedChunk(
                text=c["text"], bucket=c["bucket"], user=c["user"], score=float(s)
            )
            for s, c in ranked[:rerank_k]
        ]
    else:
        top = [
            RetrievedChunk(
                text=c["text"], bucket=c["bucket"], user=c["user"], score=1.0
            )
            for c in candidates[:rerank_k]
        ]

    return top


# ── Index builder ──────────────────────────────────────────────────────────────


def build_index(persona_path: str | Path):
    with open(persona_path) as f:
        persona = json.load(f)

    user_name = persona["profile"]["name"]
    chunks, meta = [], []

    for bucket, memories in persona["memory_buckets"].items():
        for mem in memories:
            chunks.append(mem)
            meta.append({"text": mem, "bucket": bucket, "user": user_name})

    embedder = _get_embedder()
    vecs = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    dim = vecs.shape[1]
    faiss = _get_faiss()
    index = faiss.IndexFlatIP(dim)
    index.add(vecs.astype(np.float32))
    return index, meta


def save_index(index, meta: list[dict], save_dir: str | Path) -> None:
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    _get_faiss().write_index(index, str(p / "index.faiss"))
    with open(p / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def build_all(
    memories_dir: str | Path | None = None,
    store_dir: str | Path | None = None,
) -> None:
    memories_dir = Path(memories_dir or settings.memories_dir)
    store_dir = Path(store_dir or settings.faiss_store_dir)

    for persona_file in sorted(memories_dir.glob("*.json")):
        uid = persona_file.stem
        print(f"  Building index for {uid} …")
        index, meta = build_index(persona_file)
        save_index(index, meta, store_dir / uid)
        print(f"    Saved {len(meta)} chunks → {store_dir / uid}/")
    print("\nAll indexes built.")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_all()
