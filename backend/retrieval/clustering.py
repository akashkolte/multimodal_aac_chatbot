# HDBSCAN-based semantic bucketing over BGE embeddings.
from __future__ import annotations

import json

import numpy as np

from backend.config.settings import settings
from backend.retrieval.vector_store import _get_embedder

BUCKET_LABELS = ["family", "medical", "hobbies", "daily_routine", "social"]


def cluster_persona_memories(user_id: str) -> dict[str, list[str]]:
    # Embed all memory chunks for a persona and cluster with HDBSCAN.
    import hdbscan

    memory_path = settings.memories_dir / f"{user_id}.json"
    with open(memory_path) as f:
        persona = json.load(f)

    texts, true_buckets = [], []
    for bucket, memories in persona["memory_buckets"].items():
        for mem in memories:
            texts.append(mem)
            true_buckets.append(bucket)

    embedder = _get_embedder()
    vecs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=2,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(vecs)

    clusters: dict[str, list[str]] = {}
    for text, label, _true_bucket in zip(texts, labels, true_buckets):
        key = f"cluster_{label}" if label >= 0 else "noise"
        clusters.setdefault(key, []).append(text)

    return clusters


def evaluate_bucket_alignment(user_id: str) -> dict:
    # Compare HDBSCAN clusters against hand-authored bucket labels, return purity scores.
    import hdbscan

    memory_path = settings.memories_dir / f"{user_id}.json"
    with open(memory_path) as f:
        persona = json.load(f)

    texts, true_buckets = [], []
    for bucket, memories in persona["memory_buckets"].items():
        for mem in memories:
            texts.append(mem)
            true_buckets.append(bucket)

    embedder = _get_embedder()
    vecs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2, metric="euclidean")
    pred_labels = clusterer.fit_predict(vecs)

    cluster_bucket_counts: dict[int, dict[str, int]] = {}
    for pred, true in zip(pred_labels, true_buckets):
        cluster_bucket_counts.setdefault(pred, {})
        cluster_bucket_counts[pred][true] = cluster_bucket_counts[pred].get(true, 0) + 1

    purity_scores = {}
    for cluster_id, bucket_counts in cluster_bucket_counts.items():
        total = sum(bucket_counts.values())
        dominant = max(bucket_counts.values())
        purity_scores[cluster_id] = round(dominant / total, 3)

    return {
        "n_clusters": len([k for k in purity_scores if k >= 0]),
        "n_noise": cluster_bucket_counts.get(-1, {}),
        "cluster_purity": purity_scores,
        "mean_purity": round(
            np.mean([v for k, v in purity_scores.items() if k >= 0] or [0.0]), 3
        ),
    }
