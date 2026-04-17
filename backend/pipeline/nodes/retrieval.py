# Retrieval node — dispatches each sub-intent to its pool, merges results.
from __future__ import annotations

import time

from backend.config.settings import settings
from backend.pipeline.state import PipelineState, RetrievedChunk, SubIntent
from backend.retrieval.contextual import retrieve_from_history
from backend.retrieval.vector_store import retrieve

_OPEN_DOMAIN_STUB_TEXT = (
    "(no external knowledge source wired — answer from general knowledge)"
)


def run_fast(state: PipelineState) -> dict:
    """Fast retrieval path for FRUSTRATED affect (k=2, no reranker)."""
    t0 = time.perf_counter()
    chunks = _dispatch_all(state, per_intent_k=settings.retrieval_fast_k)
    return _build_return(state, chunks, "fast", t0)


def run_full(state: PipelineState) -> dict:
    """Full retrieval path: top_k cosine matches narrowed to rerank_k."""
    t0 = time.perf_counter()
    chunks = _dispatch_all(state, per_intent_k=settings.retrieval_rerank_k)
    return _build_return(state, chunks, "full", t0)


def _dispatch_all(state: PipelineState, per_intent_k: int) -> list[RetrievedChunk]:
    route = state.get("intent_route") or {}
    sub_intents: list[SubIntent] = route.get("sub_intents") or []

    if not sub_intents:
        sub_intents = [
            {
                "type": "PERSONAL",
                "query": state["raw_query"],
                "bucket_hint": None,
                "priority": "normal",
            }
        ]

    merged: list[RetrievedChunk] = []
    for sub in sub_intents:
        kind = (sub.get("type") or "PERSONAL").upper()
        if kind == "PERSONAL":
            merged.extend(_retrieve_personal(sub, state, per_intent_k))
        elif kind == "CONTEXTUAL":
            merged.extend(_retrieve_contextual(sub, state, per_intent_k))
        elif kind == "OPEN_DOMAIN":
            merged.extend(_retrieve_open_domain(sub))
        else:
            merged.extend(_retrieve_personal(sub, state, per_intent_k))

    return _dedupe(merged)


def _retrieve_personal(
    sub: SubIntent, state: PipelineState, k: int
) -> list[RetrievedChunk]:
    priors = state["bucket_priors"]
    prior_vals = list(priors.values()) if priors else []
    priors_uniform = prior_vals and (max(prior_vals) - min(prior_vals)) < 0.05

    bucket_hint = (
        state.get("gaze_bucket")
        or sub.get("bucket_hint")
        or (_top_prior_bucket(priors) if not priors_uniform else None)
    )

    top_k = max(k, settings.retrieval_top_k) if k >= settings.retrieval_rerank_k else k
    return retrieve(
        query=sub["query"],
        user_id=state["user_id"],
        top_k=top_k,
        rerank_k=k,
        bucket_filter=bucket_hint,
    )


_CONTEXTUAL_MIN_SCORE = (
    0.5  # empirical: below this, history matches are usually spurious
)


def _retrieve_contextual(
    sub: SubIntent, state: PipelineState, k: int
) -> list[RetrievedChunk]:
    # CONTEXTUAL means "this turn leans on the recent conversation" — but the
    # persona's memories are still the primary grounding. Always pull personal
    # chunks; add contextual ones on top when the session history is relevant.
    personal_chunks = _retrieve_personal(sub, state, k)
    history = state.get("session_history") or []
    history_chunks = retrieve_from_history(query=sub["query"], history=history, top_k=k)
    relevant_history = [
        c for c in history_chunks if c["score"] >= _CONTEXTUAL_MIN_SCORE
    ]
    return personal_chunks + relevant_history


def _retrieve_open_domain(sub: SubIntent) -> list[RetrievedChunk]:
    # Intentionally a stub — web search is out of scope. See README "Intent decomposition".
    return [
        RetrievedChunk(
            text=f'{_OPEN_DOMAIN_STUB_TEXT} (sub-query: "{sub["query"]}")',
            bucket="open_domain",
            type="narrative",
            user="",
            score=0.0,
            source="open_domain",
        )
    ]


def _dedupe(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[tuple[str, str]] = set()
    out: list[RetrievedChunk] = []
    for c in chunks:
        key = (c["source"], c["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


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
