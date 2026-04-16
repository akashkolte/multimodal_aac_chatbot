# FastAPI backend — REST API for the AAC pipeline.
from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config.settings import settings
from backend.evals import compute_evals
from backend.generation.llm_client import (  # active_model used by /debug/config
    active_model,
    get_client,
)
from backend.guardrails.checks import check_input
from backend.pipeline.graph import run_pipeline
from backend.pipeline.state import PipelineState
from backend.retrieval.bucket_priors import uniform_priors
from backend.retrieval.vector_store import _get_embedder

app = FastAPI(
    title="Multimodal AAC Chatbot API",
    description="Agentic RAG pipeline for AAC persona communication",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_models_ready = False


@app.on_event("startup")
def _warmup():
    global _models_ready
    import logging
    import os

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    print("Loading models...", end=" ", flush=True)
    _get_embedder()
    get_client()
    _models_ready = True
    print("ready.")


# ── In-memory session store (replace with Redis for multi-worker deployments) ──
_sessions: dict[str, dict] = {}


# ── Request / response schemas ─────────────────────────────────────────────────


class ChatRequest(BaseModel):
    user_id: str
    query: str
    affect_override: str | None = None  # "HAPPY"|"FRUSTRATED"|"NEUTRAL"|"SURPRISED"
    gesture_tag: str | None = None
    gaze_bucket: str | None = None
    air_written_text: str | None = None


class EvalScoresResponse(BaseModel):
    groundedness: float
    hallucination_rate: float
    no_evidence: bool
    t_total_s: float
    slo_target_s: float
    slo_passed: bool
    slo_margin_s: float
    multimodal_alignment: float
    affect_alignment: float
    gesture_alignment: float
    gaze_alignment: float


class ChatResponse(BaseModel):
    user_id: str
    query: str
    response: str
    affect: str
    llm_tier: str
    llm_model: str
    retrieval_mode: str
    latency: dict
    guardrail_passed: bool
    eval_scores: EvalScoresResponse | None = None


# ── Helpers ────────────────────────────────────────────────────────────────────


def _get_or_init_session(user_id: str) -> dict:
    if user_id not in _sessions:
        try:
            with open(settings.users_json) as f:
                users = {u["id"]: u for u in json.load(f)["users"]}
        except FileNotFoundError as e:
            raise HTTPException(
                status_code=503, detail="users.json not found — run setup.sh"
            ) from e
        if user_id not in users:
            raise HTTPException(status_code=404, detail=f"User '{user_id}' not found")
        _sessions[user_id] = {
            "persona_profile": users[user_id],
            "session_history": [],
            "bucket_priors": uniform_priors(),
            "turn_id": 0,
        }
    return _sessions[user_id]


def _build_initial_state(req: ChatRequest, session: dict) -> PipelineState:
    affect_state = None
    if req.affect_override:
        affect_state = {"emotion": req.affect_override, "vector": {}, "smoothed": {}}

    session["turn_id"] += 1

    return PipelineState(
        user_id=req.user_id,
        persona_profile=session["persona_profile"],
        session_history=session["session_history"],
        turn_id=session["turn_id"],
        affect=affect_state,
        gesture_tag=req.gesture_tag,
        gaze_bucket=req.gaze_bucket,
        air_written_text=req.air_written_text,
        raw_query=req.query,
        intent_route=None,
        generation_config=None,
        retrieved_chunks=[],
        bucket_priors=session["bucket_priors"],
        retrieval_mode_used="",
        augmented_prompt=None,
        candidates=[],
        selected_response=None,
        llm_tier_used="",
        llm_model_used="",
        latency_log={
            "t_sensing": 0.0,
            "t_intent": 0.0,
            "t_retrieval": 0.0,
            "t_generation": 0.0,
            "t_total": 0.0,
        },
        run_id=None,
        guardrail_passed=True,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/health")
def health():
    return {"status": "ok", "models_ready": _models_ready}


@app.get("/debug/config")
def debug_config():
    """Return active model + key settings for the debug panel."""
    return {
        "active_llm_tier": settings.active_llm_tier,
        "active_model": active_model(),
        "thinking_mode": settings.thinking_mode,
        "embed_model": settings.embed_model,
        "retrieval_top_k": settings.retrieval_top_k,
        "retrieval_rerank_k": settings.retrieval_rerank_k,
        "fallback_latency_threshold": settings.fallback_latency_threshold,
        "slo_target_s": settings.slo_target_s,
        "num_candidates": settings.num_candidates,
    }


@app.get("/users")
def list_users():
    try:
        with open(settings.users_json) as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503, detail="users.json not found — run setup.sh"
        ) from e


@app.post("/session/reset")
def reset_session(user_id: str):
    _sessions.pop(user_id, None)
    return {"status": "reset", "user_id": user_id}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    guard = check_input(req.query)
    if not guard["allowed"]:
        return ChatResponse(
            user_id=req.user_id,
            query=req.query,
            response=guard["fallback"],
            affect="NEUTRAL",
            llm_tier="none",
            llm_model="none",
            retrieval_mode="none",
            latency={},
            guardrail_passed=False,
        )

    session = _get_or_init_session(req.user_id)
    initial_state = _build_initial_state(req, session)

    result: PipelineState = run_pipeline(initial_state)

    # Persist updated session state
    session["session_history"] = result["session_history"]
    session["bucket_priors"] = result["bucket_priors"]

    # Compute evaluation metrics
    affect_emotion = (result.get("affect") or {}).get("emotion", "NEUTRAL")
    eval_scores = compute_evals(
        response=result["selected_response"] or "",
        chunks=result.get("retrieved_chunks") or [],
        latency_log=result.get("latency_log") or {},
        affect=affect_emotion,
        gesture_tag=req.gesture_tag,
        gaze_bucket=req.gaze_bucket,
        slo_target=settings.slo_target_s,
    )

    return ChatResponse(
        user_id=req.user_id,
        query=req.query,
        response=result["selected_response"] or "",
        affect=affect_emotion,
        llm_tier=result.get("llm_tier_used", "unknown"),
        llm_model=result.get("llm_model_used", "unknown"),
        retrieval_mode=result.get("retrieval_mode_used", "unknown"),
        latency=result.get("latency_log") or {},
        guardrail_passed=result.get("guardrail_passed", True),
        eval_scores=eval_scores,
    )
