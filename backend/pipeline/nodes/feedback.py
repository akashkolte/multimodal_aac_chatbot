import json
import time
import uuid
from pathlib import Path

from backend.config.settings import settings
from backend.pipeline.state import PipelineState
from backend.retrieval.priors import BUCKETS, CHUNK_TYPES, update_weighted


def run(state: PipelineState) -> dict:
    run_id = uuid.uuid4().hex
    updated_bucket, updated_type = _update_priors(state)
    try:
        _log_to_jsonl(state, run_id, updated_bucket, updated_type)
    except Exception as exc:
        # logging never blocks the response path, but make the failure visible
        print(f"[feedback] JSONL log failed: {exc!r}")
    updated_history = _append_turn_to_history(state)

    return {
        "bucket_priors": updated_bucket,
        "type_priors": updated_type,
        "session_history": updated_history,
        "run_id": run_id,
    }


def _log_to_jsonl(
    state: PipelineState,
    run_id: str,
    bucket_priors_after: dict[str, float],
    type_priors_after: dict[str, float],
) -> None:
    logs_dir = Path(settings.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "turns.jsonl"

    latency = state.get("latency_log") or {}
    affect = (state.get("affect") or {}).get("emotion", "UNKNOWN")
    chunks = state.get("retrieved_chunks") or []
    candidates = state.get("candidates") or []

    entry = {
        "run_id": run_id,
        "ts": time.time(),
        "user_id": state["user_id"],
        "turn_id": state["turn_id"],
        "query": state["raw_query"],
        "llm_tier": state.get("llm_tier_used", "unknown"),
        "retrieval_mode": state.get("retrieval_mode_used", "unknown"),
        "affect": affect,
        "head_signal": state.get("head_signal"),
        "air_written_text": state.get("air_written_text"),
        "voice_text": state.get("voice_text"),
        "resolved_intent": state.get("resolved_intent"),
        "turnaround_triggered": state.get("turnaround_triggered", False),
        "guardrail_passed": state.get("guardrail_passed", True),
        "num_chunks": len(chunks),
        "num_personal": sum(
            1 for c in chunks if c.get("source", "personal") == "personal"
        ),
        "num_contextual": sum(1 for c in chunks if c.get("source") == "contextual"),
        "num_open_domain": sum(1 for c in chunks if c.get("source") == "open_domain"),
        "num_prior_pick": sum(1 for c in chunks if c.get("source") == "prior_pick"),
        "latency": {
            "t_sensing": latency.get("t_sensing", 0.0),
            "t_intent": latency.get("t_intent", 0.0),
            "t_retrieval": latency.get("t_retrieval", 0.0),
            "t_generation": latency.get("t_generation", 0.0),
            "t_total": latency.get("t_total", 0.0),
        },
        "response": state.get("selected_response") or "",
        "candidates": [dict(c) for c in candidates],
        "n_candidates": len(candidates),
        "bucket_priors_after": bucket_priors_after,
        "type_priors_after": type_priors_after,
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _update_priors(
    state: PipelineState,
) -> tuple[dict[str, float], dict[str, float]]:
    current_bucket = state.get("bucket_priors") or {}
    current_type = state.get("type_priors") or {}

    # Guardrail rejected the turn — no valid evidence to learn from.
    if not state.get("guardrail_passed", True):
        return current_bucket, current_type

    chunks = state.get("retrieved_chunks") or []
    personal = [c for c in chunks if c.get("source", "personal") == "personal"]
    if not personal:
        return current_bucket, current_type

    # Each retrieved chunk contributes one unit of mass to its bucket and its
    # type. Rank position already reflects relevance (top-k cosine + MMR
    # rerank), so chunk *count* per label is a cleaner signal than the raw
    # score field — especially post-MMR, where `score` mixes similarity with
    # diversity and can be negative.
    bucket_mass: dict[str, float] = {}
    type_mass: dict[str, float] = {}
    for c in personal:
        b = c.get("bucket")
        if b:
            bucket_mass[b] = bucket_mass.get(b, 0.0) + 1.0
        t = c.get("type")
        if t:
            type_mass[t] = type_mass.get(t, 0.0) + 1.0

    new_bucket = (
        update_weighted(current_bucket, bucket_mass, BUCKETS)
        if bucket_mass
        else current_bucket
    )
    new_type = (
        update_weighted(current_type, type_mass, CHUNK_TYPES)
        if type_mass
        else current_type
    )
    return new_bucket, new_type


def _append_turn_to_history(state: PipelineState) -> list[dict]:
    history = list(state.get("session_history") or [])
    history.append({"role": "partner", "content": state["raw_query"]})
    history.append(
        {"role": "aac_user", "content": state.get("selected_response") or ""}
    )
    return history
