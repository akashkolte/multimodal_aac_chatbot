import json
import threading
import time
from pathlib import Path

import torch

from backend.config.settings import settings
from backend.retrieval.vector_store import get_device, get_embedder


def _store_path(user_id: str) -> Path:
    return settings.data_dir / "pick_index" / user_id


# Guards the module-level cache. `add()` and `lookup()` can be called
# concurrently from an SSE handler and the /chat/pick POST — without this,
# a concurrent add mid-lookup could see a partially-built tensor.
_cache_lock = threading.RLock()
_cache: dict[str, tuple[torch.Tensor, list[dict]]] = {}


def _load(user_id: str) -> tuple[torch.Tensor, list[dict]]:
    with _cache_lock:
        if user_id in _cache:
            return _cache[user_id]
        p = _store_path(user_id)
        if not (p / "vectors.pt").exists():
            empty = torch.empty((0, 0), device=get_device())
            _cache[user_id] = (empty, [])
            return _cache[user_id]
        vecs = torch.load(
            p / "vectors.pt", map_location=get_device(), weights_only=True
        )
        with open(p / "entries.json") as f:
            entries = json.load(f)
        _cache[user_id] = (vecs, entries)
        return _cache[user_id]


def _persist(user_id: str, vecs: torch.Tensor, entries: list[dict]) -> None:
    p = _store_path(user_id)
    p.mkdir(parents=True, exist_ok=True)
    torch.save(vecs.detach().cpu(), p / "vectors.pt")
    with open(p / "entries.json", "w") as f:
        json.dump(entries, f, indent=2)


def lookup(query: str, user_id: str, threshold: float = 0.85) -> dict | None:
    # Snapshot the (vecs, entries) tuple under the lock so a concurrent add()
    # can't swap it out mid-search. Read-only work on the snapshot is safe.
    with _cache_lock:
        vecs, entries = _load(user_id)
    if vecs.numel() == 0 or not entries:
        return None
    embedder = get_embedder()
    q = embedder.encode(
        [query],
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=get_device(),
    )[0]
    scores = vecs @ q
    top_score, top_idx = torch.max(scores, dim=0)
    score = float(top_score)
    if score < threshold:
        return None
    hit = dict(entries[int(top_idx)])
    hit["match_score"] = score
    return hit


def add(
    query: str,
    user_id: str,
    strategy: str,
    picked_text: str,
    picked_buckets: list[str] | None = None,
) -> None:
    embedder = get_embedder()
    q = embedder.encode(
        [query],
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=get_device(),
    )  # (1, D)
    # The whole read-modify-write is locked so two concurrent adds can't
    # both read the same `vecs`, each concat their own vector, and clobber
    # each other on writeback.
    with _cache_lock:
        vecs, entries = _load(user_id)
        new_vecs = q if vecs.numel() == 0 else torch.cat([vecs, q], dim=0)
        new_entries = list(entries) + [
            {
                "query": query,
                "strategy": strategy,
                "picked_text": picked_text,
                "picked_buckets": picked_buckets or [],
                "ts": time.time(),
            }
        ]
        _cache[user_id] = (new_vecs, new_entries)
        _persist(user_id, new_vecs, new_entries)


def bucket_pick_counts(user_id: str) -> dict[str, float]:
    """Cumulative pick counts per bucket for this user.

    Each pick contributes 1.0 mass split evenly across the buckets grounding
    the picked candidate. Used by retrieval to bias bucket priors toward
    memories the user has historically preferred.
    """
    with _cache_lock:
        _, entries = _load(user_id)
        entries_snapshot = list(entries)
    counts: dict[str, float] = {}
    for e in entries_snapshot:
        buckets = [b for b in (e.get("picked_buckets") or []) if b]
        if not buckets:
            continue
        share = 1.0 / len(buckets)
        for b in buckets:
            counts[b] = counts.get(b, 0.0) + share
    return counts
