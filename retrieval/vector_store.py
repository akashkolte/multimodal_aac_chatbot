"""
FAISS-backed dense retrieval with BGE embeddings and cross-encoder reranking.

Models are lazy-loaded on first use (safe for FastAPI / LangGraph workers).

NOTE: The FAISS indexes in data/faiss_store/ must be built with BGE embeddings.
      Run `python -m retrieval.vector_store` to rebuild all persona indexes.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from config.settings import settings
from pipeline.state import RetrievedChunk

# ── Lazy model singletons ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


@lru_cache(maxsize=1)
def _get_reranker() -> CrossEncoder:
    return CrossEncoder(settings.rerank_model)


# ── Index cache (one FAISS index per user_id) ─────────────────────────────────

_index_cache: dict[str, tuple[faiss.Index, list[dict]]] = {}


def load_index(user_id: str) -> tuple[faiss.Index, list[dict]]:
    if user_id not in _index_cache:
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
    debug: bool = False,
) -> list[RetrievedChunk]:
    """
    Two-stage retrieval:
      1. BGE-small-en-v1.5 bi-encoder → FAISS IndexFlatIP (cosine similarity)
      2. BGE-reranker-v2-m3 cross-encoder reranking (multilingual, skippable)

    Args:
        query:         Partner's text query.
        user_id:       Persona identifier (e.g. "mia_chen").
        top_k:         Number of candidates from FAISS before reranking.
        rerank_k:      Final number of chunks returned after reranking.
        bucket_filter: If set, restrict candidates to this memory bucket.
        use_reranker:  False for the FRUSTRATED fast path.
        debug:         Return timing breakdown alongside results.
    """
    embedder = _get_embedder()
    index, meta = load_index(user_id)

    t0 = time.perf_counter()
    q_vec = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, idxs = index.search(q_vec, top_k)
    t_faiss = time.perf_counter() - t0

    candidates = [meta[i] for i in idxs[0] if i < len(meta)]

    if bucket_filter:
        filtered = [c for c in candidates if c["bucket"] == bucket_filter]
        candidates = filtered if filtered else candidates   # fallback: all buckets

    t0 = time.perf_counter()
    if use_reranker and len(candidates) > 1:
        reranker = _get_reranker()
        pairs = [(query, c["text"]) for c in candidates]
        ce_scores = reranker.predict(pairs)
        ranked = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
        top = [
            RetrievedChunk(text=c["text"], bucket=c["bucket"], user=c["user"], score=float(s))
            for s, c in ranked[:rerank_k]
        ]
    else:
        top = [
            RetrievedChunk(text=c["text"], bucket=c["bucket"], user=c["user"], score=1.0)
            for c in candidates[:rerank_k]
        ]
    t_rerank = time.perf_counter() - t0

    if debug:
        return top, {"t_embed": t_embed, "t_faiss": t_faiss, "t_rerank": t_rerank}
    return top


# ── Index builder ──────────────────────────────────────────────────────────────

def build_index(persona_path: str | Path) -> tuple[faiss.Index, list[dict]]:
    """Embed all memory chunks for a persona and build a FAISS IndexFlatIP."""
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
    index = faiss.IndexFlatIP(dim)
    index.add(vecs.astype(np.float32))
    return index, meta


def save_index(index: faiss.Index, meta: list[dict], save_dir: str | Path) -> None:
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p / "index.faiss"))
    with open(p / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def build_all(
    memories_dir: str | Path | None = None,
    store_dir: str | Path | None = None,
) -> None:
    """Rebuild FAISS indexes for all personas using the configured BGE embedder."""
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
