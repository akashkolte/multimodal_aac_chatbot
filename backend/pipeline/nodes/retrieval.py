# Retrieval node — run_fast (FRUSTRATED) and run_full paths.
from __future__ import annotations

import time

from backend.config.settings import settings
from backend.pipeline.state import PipelineState, RetrievedChunk
from backend.retrieval.vector_store import retrieve


def run_fast(state: PipelineState) -> dict:
    """Fast retrieval path for FRUSTRATED affect (k=2, no reranker)."""
    t0 = time.perf_counter()

    priors = state["bucket_priors"]
    prior_vals = list(priors.values()) if priors else []
    priors_uniform = prior_vals and (max(prior_vals) - min(prior_vals)) < 0.05
    bucket_hint = (
        state.get("gaze_bucket")
        if priors_uniform and state.get("gaze_bucket")
        else _top_prior_bucket(priors)
    )
    chunks = retrieve(
        query=state["raw_query"],
        user_id=state["user_id"],
        top_k=settings.retrieval_fast_k,
        rerank_k=settings.retrieval_fast_k,
        bucket_filter=bucket_hint,
    )

    return _build_return(state, chunks, "fast", t0)


def run_full(state: PipelineState) -> dict:
    """Full retrieval path: top_k cosine matches narrowed to rerank_k."""
    t0 = time.perf_counter()

    # Prefer gaze hint > intent bucket hint > None
    route = state.get("intent_route") or {}
    sub_intents = route.get("sub_intents", [])
    bucket_hint = state.get("gaze_bucket") or next(
        (si.get("bucket_hint") for si in sub_intents if si.get("bucket_hint")), None
    )

    chunks = retrieve(
        query=state["raw_query"],
        user_id=state["user_id"],
        top_k=settings.retrieval_top_k,
        rerank_k=settings.retrieval_rerank_k,
        bucket_filter=bucket_hint,
    )

    return _build_return(state, chunks, "full", t0)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _top_prior_bucket(priors: dict[str, float]) -> str | None:
    if not priors:
        return None
    return max(priors, key=priors.get)


def _build_return(
    state: PipelineState,
    chunks: list[RetrievedChunk],
    mode: str,
    t0: float,
) -> dict:
    t_retrieval = time.perf_counter() - t0

    latency_log = dict(state.get("latency_log") or {})
    latency_log["t_retrieval"] = round(t_retrieval, 4)

    return {
        "retrieved_chunks": chunks,
        "retrieval_mode_used": mode,
        "latency_log": latency_log,
    }
