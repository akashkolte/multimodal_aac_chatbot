"""
L3 — Semantic Bucketing & Retrieval node.

Two entry points:
  run_fast  — FRUSTRATED affect: k=2, single bucket, no reranking
  run_full  — standard: k=5, optional bucket hint, BGE cross-encoder reranking

Also exports the conditional edge function used by graph.py.
"""
from __future__ import annotations

import time

from config.settings import settings
from pipeline.state import PipelineState, RetrievedChunk
from retrieval.vector_store import retrieve
from retrieval.bucket_priors import update_priors


def run_fast(state: PipelineState) -> dict:
    """Fast retrieval path for FRUSTRATED affect (k=2, no reranker)."""
    t0 = time.perf_counter()

    bucket_hint = _top_prior_bucket(state["bucket_priors"])
    chunks = retrieve(
        query=state["raw_query"],
        user_id=state["user_id"],
        top_k=settings.retrieval_fast_k,
        rerank_k=settings.retrieval_fast_k,
        bucket_filter=bucket_hint,
        use_reranker=False,
    )

    return _build_return(state, chunks, "fast", t0)


def run_full(state: PipelineState) -> dict:
    """Full retrieval path with BGE cross-encoder reranking."""
    t0 = time.perf_counter()

    # Prefer gaze hint > intent bucket hint > None
    route = state.get("intent_route") or {}
    sub_intents = route.get("sub_intents", [])
    bucket_hint = (
        state.get("gaze_bucket")
        or next((si.get("bucket_hint") for si in sub_intents if si.get("bucket_hint")), None)
    )

    chunks = retrieve(
        query=state["raw_query"],
        user_id=state["user_id"],
        top_k=settings.retrieval_top_k,
        rerank_k=settings.retrieval_rerank_k,
        bucket_filter=bucket_hint,
        use_reranker=True,
    )

    return _build_return(state, chunks, "full", t0)


# def route_by_affect(state: PipelineState) -> str:
#     """Conditional edge function — called by graph.py after the intent node."""
#     emotion = (state.get("affect") or {}).get("emotion", "NEUTRAL")
#     return "fast" if emotion == "FRUSTRATED" else "full"


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
