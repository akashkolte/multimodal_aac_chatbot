"""
L5 — Feedback Loop node.

After a response is accepted:
  1. Log the full turn to MLflow (latency, metrics, prompt version, tier used)
  2. Update session-level Bayesian bucket priors
  3. Append the accepted turn to session history

Rejected candidates are also logged for offline analysis.
"""
from __future__ import annotations

import json
import time

import mlflow

from config.settings import settings
from pipeline.state import PipelineState
from retrieval.bucket_priors import update_priors


def run(state: PipelineState) -> dict:
    t0 = time.perf_counter()

    mlflow_run_id = _log_to_mlflow(state)
    updated_priors = _update_bucket_priors(state)
    updated_history = _append_turn_to_history(state)

    return {
        "bucket_priors": updated_priors,
        "session_history": updated_history,
        "mlflow_run_id": mlflow_run_id,
    }


# ── MLflow logging ─────────────────────────────────────────────────────────────

def _log_to_mlflow(state: PipelineState) -> str:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)

    latency = state.get("latency_log") or {}
    affect = (state.get("affect") or {}).get("emotion", "UNKNOWN")

    with mlflow.start_run(run_name=f"turn-{state['turn_id']}") as run:
        mlflow.log_params({
            "user_id":         state["user_id"],
            "turn_id":         state["turn_id"],
            "llm_tier":        state.get("llm_tier_used", "unknown"),
            "retrieval_mode":  state.get("retrieval_mode_used", "unknown"),
            "affect":          affect,
            "guardrail_passed": state.get("guardrail_passed", True),
        })
        mlflow.log_metrics({
            "t_sensing":    latency.get("t_sensing",    0.0),
            "t_intent":     latency.get("t_intent",     0.0),
            "t_retrieval":  latency.get("t_retrieval",  0.0),
            "t_generation": latency.get("t_generation", 0.0),
            "t_total":      latency.get("t_total",      0.0),
            "num_chunks":   float(len(state.get("retrieved_chunks") or [])),
        })

        # Log the selected response as artifact text for qualitative review
        mlflow.log_text(
            state.get("selected_response") or "",
            f"responses/turn_{state['turn_id']}.txt",
        )

        return run.info.run_id


# ── Bayesian bucket prior update ───────────────────────────────────────────────

def _update_bucket_priors(state: PipelineState) -> dict[str, float]:
    chunks = state.get("retrieved_chunks") or []
    if not chunks:
        return state.get("bucket_priors") or {}

    # Which bucket sourced the accepted response?
    top_bucket = chunks[0].get("bucket")
    if not top_bucket:
        return state.get("bucket_priors") or {}

    return update_priors(
        priors=state.get("bucket_priors") or {},
        accepted_bucket=top_bucket,
    )


# ── Session history append ─────────────────────────────────────────────────────

def _append_turn_to_history(state: PipelineState) -> list[dict]:
    """Returns a single-element list; LangGraph's Annotated[list, add] merges it."""
    return [
        {"role": "partner",  "content": state["raw_query"]},
        {"role": "aac_user", "content": state.get("selected_response") or ""},
    ]
