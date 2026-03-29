import json
import faiss
import numpy as np
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Models (loaded once) ──────────────────────────────────────────────────────
embedder  = SentenceTransformer("all-MiniLM-L6-v2")
reranker  = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Build index from a persona JSON ──────────────────────────────────────────
def build_index(persona_path: str):
    """
    Reads a bucketed persona JSON, embeds every memory chunk,
    stores them in a FAISS flat index.

    Returns: (index, metadata_list)
    metadata_list[i] = {"text": ..., "bucket": ..., "user": ...}
    """
    with open(persona_path) as f:
        persona = json.load(f)

    user_name = persona["profile"]["name"]
    chunks, meta = [], []

    for bucket, memories in persona["memory_buckets"].items():
        for mem in memories:
            chunks.append(mem)
            meta.append({"text": mem, "bucket": bucket, "user": user_name})

    vecs = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)

    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product == cosine sim (vecs normalised)
    index.add(vecs)

    return index, meta


def save_index(index, meta, save_dir: str):
    """Persist index + metadata so you don't rebuild every run."""
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(p / "index.faiss"))
    with open(p / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_index(save_dir: str):
    p = Path(save_dir)
    index = faiss.read_index(str(p / "index.faiss"))
    with open(p / "meta.json") as f:
        meta = json.load(f)
    return index, meta


# ── Retrieve + rerank ─────────────────────────────────────────────────────────
def retrieve(query: str, index, meta, top_k=10, rerank_k=3, bucket_filter=None, debug=False):
    """
    1. Embed query → FAISS top_k
    2. (Optional) filter by bucket
    3. Cross-encoder rerank → return top rerank_k

    bucket_filter: e.g. "family" or None for all buckets
    """
    t_embed_start = time.perf_counter()
    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    t_embed = time.perf_counter() - t_embed_start

    t_faiss_start = time.perf_counter()
    scores, idxs = index.search(q_vec, top_k)
    t_faiss = time.perf_counter() - t_faiss_start

    candidates = [meta[i] for i in idxs[0]]

    # Optional bucket filter (L2 controller hints which bucket to prioritise)
    if bucket_filter:
        filtered = [c for c in candidates if c["bucket"] == bucket_filter]
        candidates = filtered if filtered else candidates   # fallback to all

    # Cross-encoder reranking
    t_rerank_start = time.perf_counter()
    pairs   = [(query, c["text"]) for c in candidates]
    ce_scores = reranker.predict(pairs)

    ranked  = sorted(zip(ce_scores, candidates), key=lambda x: x[0], reverse=True)
    top     = [c for _, c in ranked[:rerank_k]]
    t_rerank = time.perf_counter() - t_rerank_start

    if debug:
        timings = {
            "retrieve_embed": t_embed,
            "retrieve_faiss": t_faiss,
            "retrieve_rerank": t_rerank,
            "retrieve_total": t_embed + t_faiss + t_rerank,
        }
        return top, timings

    return top   # list of {"text", "bucket", "user"}


# ── Quick test ────────────────────────────────────────────────────────────────
# ── Build indexes for ALL personas ────────────────────────────────────────────
def build_all(memories_dir="data/memories", store_dir="data/faiss_store"):
    for pf in Path(memories_dir).glob("*.json"):
        uid = pf.stem
        print(f"  Building index for {uid}...")
        index, meta = build_index(str(pf))
        save_index(index, meta, f"{store_dir}/{uid}")
        print(f"    Saved {len(meta)} chunks -> data/faiss_store/{uid}/")
    print("\n  Done.")

def test_retrieve(uid="mia_chen", query="what do you like to do on weekends?"):
    index, meta = load_index(f"data/faiss_store/{uid}")
    results = retrieve(query, index, meta, top_k=10, rerank_k=3)
    print(f"\nQuery: {query}\nPersona: {uid}\n")
    for r in results:
        print(f"  [{r['bucket']}] {r['text']}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_all()
    test_retrieve()