# Retrieval node — dispatches each sub-intent to its pool, merges results.
import time

import torch

from backend.config.settings import settings
from backend.pipeline.intent_kind import is_present_state_only
from backend.pipeline.state import PipelineState, RetrievedChunk, SubIntent
from backend.retrieval import pick_index
from backend.retrieval.contextual import retrieve_from_history
from backend.retrieval.priors import BUCKETS
from backend.retrieval.reranker import build_context_vector, mmr_rerank
from backend.retrieval.vector_store import get_device, get_embedder, retrieve

# Weight of the pick-history bucket prior, relative to the session bucket prior.
# 0.3 means: a user who always picks "family" over "medical" gets a noticeable
# but not overwhelming nudge — session-in-progress signals still dominate.
_PICK_PRIOR_WEIGHT = 0.3

_OPEN_DOMAIN_STUB_TEXT = (
    "(no external knowledge source wired — answer from general knowledge)"
)


def run_fast(state: PipelineState) -> dict:
    """Fast retrieval path for FRUSTRATED affect (k=2, MMR-only rerank)."""
    t0 = time.perf_counter()
    if is_present_state_only(state.get("intent_route")):
        return _build_return(state, [], "skipped_present_state", t0, 0.0)
    session_priors = state.get("bucket_priors")
    _blend_pick_history_into_priors(state)
    final_k = settings.retrieval_fast_k
    pool_k = settings.rerank_fast_pool_k
    chunks, t_rerank = _dispatch_all(state, pool_k=pool_k, final_k=final_k)
    if session_priors is not None:
        state["bucket_priors"] = session_priors  # blend was transient
    chunks = _prepend_prior_pick(state, chunks)
    return _build_return(state, chunks, "fast", t0, t_rerank)


def run_full(state: PipelineState) -> dict:
    """Full retrieval path: wide cosine pool reranked by MMR + conversation context."""
    t0 = time.perf_counter()
    if is_present_state_only(state.get("intent_route")):
        return _build_return(state, [], "skipped_present_state", t0, 0.0)
    session_priors = state.get("bucket_priors")
    _blend_pick_history_into_priors(state)
    final_k = settings.retrieval_rerank_k
    pool_k = settings.rerank_pool_k
    chunks, t_rerank = _dispatch_all(state, pool_k=pool_k, final_k=final_k)
    if session_priors is not None:
        state["bucket_priors"] = session_priors  # blend was transient
    chunks = _prepend_prior_pick(state, chunks)
    return _build_return(state, chunks, "full", t0, t_rerank)


def _blend_pick_history_into_priors(state: PipelineState) -> None:
    """Mix cumulative bucket-pick counts into this turn's bucket_priors.

    Mutates state in-place. Session priors still dominate; pick history adds
    a small, steady bias toward buckets the user has historically picked.
    """
    try:
        counts = pick_index.bucket_pick_counts(state["user_id"])
    except Exception as exc:
        print(f"[retrieval] pick_index.bucket_pick_counts failed: {exc!r}")
        return
    if not counts:
        return
    total = sum(counts.values())
    if total <= 0:
        return
    pick_dist = {b: counts.get(b, 0.0) / total for b in BUCKETS}
    session_priors = state.get("bucket_priors") or {}
    if not session_priors:
        session_priors = {b: 1.0 / len(BUCKETS) for b in BUCKETS}
    blended = {
        b: (1 - _PICK_PRIOR_WEIGHT) * session_priors.get(b, 0.0)
        + _PICK_PRIOR_WEIGHT * pick_dist[b]
        for b in BUCKETS
    }
    s = sum(blended.values())
    if s > 0:
        blended = {b: v / s for b, v in blended.items()}
    state["bucket_priors"] = blended


def _prepend_prior_pick(
    state: PipelineState, chunks: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """On a side-index hit, surface the previously-picked text as a special
    chunk the LLM sees in its grounding block. Not deduped against personal
    chunks — the prior pick is phrased in the persona's voice and is useful
    even when similar memories are present.
    """
    try:
        hit = pick_index.lookup(
            query=state["raw_query"], user_id=state["user_id"], threshold=0.85
        )
    except Exception as exc:
        print(f"[retrieval] pick_index.lookup failed: {exc!r}")
        return chunks
    if not hit:
        return chunks
    text = (hit.get("picked_text") or "").strip()
    if not text:
        return chunks
    # Avoid injecting an identical chunk twice (the side_index-strategy
    # candidate in the planner handles that path separately).
    if any(c.get("text") == text for c in chunks):
        return chunks
    prior = RetrievedChunk(
        text=text,
        bucket="prior_pick",
        type="narrative",
        user="",
        score=float(hit.get("match_score", 0.0)),
        source="prior_pick",
    )
    return [prior] + list(chunks)


def _dispatch_all(
    state: PipelineState, pool_k: int, final_k: int
) -> tuple[list[RetrievedChunk], float]:
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

    rerankable: list[tuple[RetrievedChunk, torch.Tensor | None]] = []
    pinned: list[RetrievedChunk] = []

    # When rerank is off we'd just truncate to final_k anyway, so don't fetch the wider pool.
    fetch_k = pool_k if settings.rerank_enabled else final_k

    for sub in sub_intents:
        kind = (sub.get("type") or "PERSONAL").upper()
        if kind == "PERSONAL":
            rerankable.extend(_retrieve_personal(sub, state, fetch_k))
        elif kind == "CONTEXTUAL":
            rerankable.extend(_retrieve_personal(sub, state, fetch_k))
            for c in _retrieve_contextual_history(sub, state, final_k):
                rerankable.append((c, None))
        elif kind == "OPEN_DOMAIN":
            pinned.extend(_retrieve_open_domain(sub))
        elif kind == "PRESENT_STATE":
            # PRESENT_STATE is grounded in the affect signal, not memory.
            # In a pure-present-state route the run_fast/run_full early skip
            # already short-circuits us; in a mixed route we just contribute
            # nothing here so the planner doesn't see misleading chunks.
            continue
        else:
            rerankable.extend(_retrieve_personal(sub, state, fetch_k))

    rerankable = _dedupe_with_vecs(rerankable)

    t_rerank = 0.0
    if not settings.rerank_enabled or not rerankable:
        chunks = [c for c, _ in rerankable[:final_k]]
    else:
        t1 = time.perf_counter()
        chunks = _rerank_merged(state, rerankable, final_k)
        t_rerank = time.perf_counter() - t1

    return chunks + pinned, t_rerank


def _retrieve_personal(
    sub: SubIntent, state: PipelineState, k: int
) -> list[tuple[RetrievedChunk, torch.Tensor | None]]:
    # Gaze fixation is an explicit user signal — hard filter. Session priors
    # (bucket + type) are applied as soft weights inside vector_store.retrieve().
    hard_filter = state.get("gaze_bucket")
    bucket_priors = state.get("bucket_priors")
    type_priors = state.get("type_priors")

    if not settings.rerank_enabled:
        chunks = retrieve(
            query=sub["query"],
            user_id=state["user_id"],
            top_k=k,
            rerank_k=k,
            bucket_filter=hard_filter,
            bucket_priors=bucket_priors,
            type_priors=type_priors,
        )
        return [(c, None) for c in chunks]

    chunks, vecs = retrieve(
        query=sub["query"],
        user_id=state["user_id"],
        top_k=k,
        rerank_k=k,
        bucket_filter=hard_filter,
        bucket_priors=bucket_priors,
        type_priors=type_priors,
        return_vectors=True,
    )
    return [(chunks[i], vecs[i]) for i in range(len(chunks))]


_CONTEXTUAL_MIN_SCORE = (
    0.5  # empirical: below this, history matches are usually spurious
)


def _retrieve_contextual_history(
    sub: SubIntent, state: PipelineState, k: int
) -> list[RetrievedChunk]:
    history = state.get("session_history") or []
    history_chunks = retrieve_from_history(query=sub["query"], history=history, top_k=k)
    return [c for c in history_chunks if c["score"] >= _CONTEXTUAL_MIN_SCORE]


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


def _dedupe_with_vecs(
    items: list[tuple[RetrievedChunk, torch.Tensor | None]],
) -> list[tuple[RetrievedChunk, torch.Tensor | None]]:
    seen: set[tuple[str, str]] = set()
    out: list[tuple[RetrievedChunk, torch.Tensor | None]] = []
    for c, v in items:
        key = (c["source"], c["text"])
        if key in seen:
            continue
        seen.add(key)
        out.append((c, v))
    return out


def _rerank_merged(
    state: PipelineState,
    items: list[tuple[RetrievedChunk, torch.Tensor | None]],
    final_k: int,
) -> list[RetrievedChunk]:
    missing_idxs = [i for i, (_, v) in enumerate(items) if v is None]
    if missing_idxs:
        embedder = get_embedder()
        encoded = embedder.encode(
            [items[i][0]["text"] for i in missing_idxs],
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=get_device(),
        )
        vecs: list[torch.Tensor] = []
        encoded_iter = iter(encoded)
        for _, v in items:
            vecs.append(next(encoded_iter) if v is None else v)
    else:
        vecs = [v for _, v in items]  # type: ignore[misc]

    candidate_vecs = torch.stack(vecs)  # (N, D)
    candidate_chunks = [c for c, _ in items]

    query_vec = build_context_vector(
        raw_query=state["raw_query"],
        history=state.get("session_history"),
        last_n_turns=settings.rerank_history_turns,
        weight_current=settings.rerank_query_weight,
    )

    return mmr_rerank(
        query_vec=query_vec,
        candidate_vecs=candidate_vecs,
        candidate_chunks=candidate_chunks,
        top_k=final_k,
        lambda_=settings.rerank_lambda,
    )


def _build_return(
    state: PipelineState,
    chunks: list[RetrievedChunk],
    mode: str,
    t0: float,
    t_rerank: float,
) -> dict:
    t_retrieval = time.perf_counter() - t0

    latency_log = dict(state.get("latency_log") or {})
    latency_log["t_retrieval"] = round(t_retrieval, 4)
    latency_log["t_rerank"] = round(t_rerank, 4)

    return {
        "retrieved_chunks": chunks,
        "retrieval_mode_used": mode,
        "latency_log": latency_log,
    }
