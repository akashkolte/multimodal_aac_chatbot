# Feedback node — JSONL turn logging, bucket prior update, history append.
import json
import time
import uuid
from pathlib import Path

from backend.config.settings import settings
from backend.pipeline.state import PipelineState
from backend.retrieval.bucket_priors import update_priors


def run(state: PipelineState) -> dict:
    run_id = uuid.uuid4().hex
    try:
        _log_to_jsonl(state, run_id)
    except Exception as exc:
        # logging never blocks the response path, but make the failure visible
        print(f"[feedback] JSONL log failed: {exc!r}")
    updated_priors = _update_bucket_priors(state)
    updated_history = _append_turn_to_history(state)

    return {
        "bucket_priors": updated_priors,
        "session_history": updated_history,
        "run_id": run_id,
    }


def _log_to_jsonl(state: PipelineState, run_id: str) -> None:
    logs_dir = Path(settings.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "turns.jsonl"

    latency = state.get("latency_log") or {}
    affect = (state.get("affect") or {}).get("emotion", "UNKNOWN")
    chunks = state.get("retrieved_chunks") or []

    entry = {
        "run_id": run_id,
        "ts": time.time(),
        "user_id": state["user_id"],
        "turn_id": state["turn_id"],
        "llm_tier": state.get("llm_tier_used", "unknown"),
        "retrieval_mode": state.get("retrieval_mode_used", "unknown"),
        "affect": affect,
        "head_signal": state.get("head_signal"),
        "turnaround_triggered": state.get("turnaround_triggered", False),
        "guardrail_passed": state.get("guardrail_passed", True),
        "num_chunks": len(chunks),
        "num_personal": sum(
            1 for c in chunks if c.get("source", "personal") == "personal"
        ),
        "num_contextual": sum(1 for c in chunks if c.get("source") == "contextual"),
        "num_open_domain": sum(1 for c in chunks if c.get("source") == "open_domain"),
        "latency": {
            "t_sensing": latency.get("t_sensing", 0.0),
            "t_intent": latency.get("t_intent", 0.0),
            "t_retrieval": latency.get("t_retrieval", 0.0),
            "t_generation": latency.get("t_generation", 0.0),
            "t_total": latency.get("t_total", 0.0),
        },
        "response": state.get("selected_response") or "",
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _update_bucket_priors(state: PipelineState) -> dict[str, float]:
    chunks = state.get("retrieved_chunks") or []
    personal = [c for c in chunks if c.get("source", "personal") == "personal"]
    if not personal:
        return state.get("bucket_priors") or {}

    top_bucket = personal[0].get("bucket")
    if not top_bucket:
        return state.get("bucket_priors") or {}

    return update_priors(
        priors=state.get("bucket_priors") or {},
        accepted_bucket=top_bucket,
    )


def _append_turn_to_history(state: PipelineState) -> list[dict]:
    history = list(state.get("session_history") or [])
    history.append({"role": "partner", "content": state["raw_query"]})
    history.append(
        {"role": "aac_user", "content": state.get("selected_response") or ""}
    )
    return history
