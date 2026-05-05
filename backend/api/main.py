# FastAPI backend — REST API for the AAC pipeline.
import json
import logging
import re
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config.settings import settings
from backend.evals import compute_evals
from backend.generation.llm_client import (  # active_model used by /debug/config
    active_model,
    get_client,
)
from backend.guardrails.checks import check_input
from backend.pipeline.graph import choose_planner_tier, run_pipeline, run_until_planner
from backend.pipeline.intent_kind import classify_intent_kind
from backend.pipeline.nodes import feedback as feedback_node
from backend.pipeline.nodes import planner as planner_node
from backend.pipeline.state import PipelineState
from backend.retrieval import pick_index
from backend.retrieval.priors import BUCKETS, CHUNK_TYPES, uniform
from backend.retrieval.vector_store import _get_embedder, retrieve

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

_log = logging.getLogger(__name__)
_models_ready = False
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"


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

# Eval scores keyed by run_id, filled by a BackgroundTask after /chat returns
# so the UI can render the response immediately and poll GET /evals/{run_id}.
# Multi-worker deploys should swap this (and _sessions) for Redis.
_EVAL_FAILED: dict = {"_failed": True}
_eval_results: OrderedDict[str, dict] = OrderedDict()
_eval_lock = threading.Lock()
_EVAL_RESULTS_MAX = 200


def _remember_eval(run_id: str, scores: dict | None) -> None:
    value = scores if scores else _EVAL_FAILED
    with _eval_lock:
        _eval_results[run_id] = value
        _eval_results.move_to_end(run_id)
        while len(_eval_results) > _EVAL_RESULTS_MAX:
            _eval_results.popitem(last=False)


def _reserve_eval_slot(run_id: str) -> None:
    """Mark a run_id as in-flight so /evals can report 'pending' vs 'unknown'."""
    with _eval_lock:
        if run_id not in _eval_results:
            _eval_results[run_id] = {}  # empty dict = pending
            _eval_results.move_to_end(run_id)
            while len(_eval_results) > _EVAL_RESULTS_MAX:
                _eval_results.popitem(last=False)


# ── Request / response schemas ─────────────────────────────────────────────────


class ResolvedIntent(BaseModel):
    text: str
    source: str  # voice_only | air_only | agree | conflict_air | conflict_voice | none
    voice_text: str | None = None
    air_text: str | None = None


class ChatRequest(BaseModel):
    user_id: str
    query: str
    affect_override: str | None = None  # "HAPPY"|"FRUSTRATED"|"NEUTRAL"|"SURPRISED"
    gesture_tag: str | None = None
    gaze_bucket: str | None = None
    air_written_text: str | None = None
    head_signal: str | None = None  # "HEAD_SHAKE"|"HEAD_NOD_DISSATISFIED"
    voice_text: str | None = None
    resolved_intent: ResolvedIntent | None = None


class TurnaroundRequest(BaseModel):
    user_id: str
    turn_id: int | None = None  # optional guard against stale turnaround calls
    head_signal: str | None = None


class CandidateOut(BaseModel):
    text: str
    strategy: str
    grounded_buckets: list[str] = []


class ChatResponse(BaseModel):
    user_id: str
    query: str
    response: str
    candidates: list[CandidateOut] = []
    affect: str
    llm_tier: str
    llm_model: str
    retrieval_mode: str
    latency: dict
    guardrail_passed: bool
    run_id: str | None = None
    turn_id: int
    eval_scores: dict | None = None


class PickRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=64, pattern=_ID_PATTERN)
    user_id: str = Field(min_length=1, max_length=64, pattern=_ID_PATTERN)
    picked_idx: int = Field(ge=0, le=10)


class RegenerateRequest(BaseModel):
    user_id: str
    turn_id: int | None = None
    rejected_texts: list[str] = Field(default_factory=list, max_length=20)


class RatingRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=64, pattern=_ID_PATTERN)
    user_id: str = Field(min_length=1, max_length=64, pattern=_ID_PATTERN)
    authenticity: int = Field(ge=1, le=5)
    rater_id: str = Field(default="anonymous", max_length=64, pattern=_ID_PATTERN)
    notes: str | None = Field(default=None, max_length=500)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _load_persona_profile(user_id: str) -> dict:
    memories_path = settings.memories_dir / f"{user_id}.json"
    try:
        with open(memories_path) as f:
            persona = json.load(f)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Persona file not found: {memories_path}",
        ) from e
    return persona["profile"]


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
            "persona_profile": _load_persona_profile(user_id),
            "session_history": [],
            "bucket_priors": uniform(BUCKETS),
            "type_priors": uniform(CHUNK_TYPES),
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
        head_signal=req.head_signal,
        voice_text=req.voice_text,
        resolved_intent=(
            req.resolved_intent.model_dump() if req.resolved_intent else None
        ),
        turnaround_triggered=False,
        raw_query=req.query,
        intent_route=None,
        generation_config=None,
        retrieved_chunks=[],
        bucket_priors=session["bucket_priors"],
        type_priors=session["type_priors"],
        retrieval_mode_used="",
        augmented_prompt=None,
        candidates=[],
        rejected_candidates=[],
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


def _re_retrieve_excluding(
    query: str,
    user_id: str,
    rejected_chunks: list[dict],
) -> list[dict] | None:
    """Pull fresh chunks for a turnaround, excluding the bucket and exact texts
    of the rejected chunks.

    Returns:
        - list of chunks (passing min-score floor) when re-retrieval improved
          on the rejected set
        - None when re-retrieval should not be used (no signal, all dropped by
          dedupe, or all below score floor) — caller should keep original chunks
    """
    if not rejected_chunks:
        return None
    rejected_bucket = rejected_chunks[0].get("bucket")
    rejected_texts = {c.get("text") for c in rejected_chunks if c.get("text")}
    if not rejected_bucket:
        return None

    try:
        # Pull a wider net (top_k * 2) so dedupe + bucket-exclusion still leaves
        # enough candidates to fill rerank_k.
        fresh = retrieve(
            query=query,
            user_id=user_id,
            top_k=settings.retrieval_top_k * 2,
            rerank_k=settings.retrieval_top_k * 2,
            bucket_filter=None,
        )
    except Exception as exc:
        _log.warning("turnaround re-retrieval failed: %r", exc)
        return None

    filtered = [
        c
        for c in fresh
        if c.get("bucket") != rejected_bucket
        and c.get("text") not in rejected_texts
        and float(c.get("score", 0.0)) >= settings.turnaround_min_score
    ]
    if not filtered:
        _log.info(
            "turnaround re-retrieval found no chunks above score floor %.2f — "
            "keeping original chunks",
            settings.turnaround_min_score,
        )
        return None
    return filtered[: settings.retrieval_rerank_k]


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


def _compute_and_persist_evals(
    run_id: str | None,
    user_id: str,
    turn_id: int,
    response: str,
    chunks: list[dict],
    latency_log: dict,
    affect: str,
    gesture_tag: str | None,
    gaze_bucket: str | None,
) -> dict | None:
    if not settings.evals_enabled or not run_id:
        return None
    try:
        scores = compute_evals(
            response=response,
            chunks=chunks,
            latency_log=latency_log,
            affect=affect,
            gesture_tag=gesture_tag,
            gaze_bucket=gaze_bucket,
            slo_target=settings.slo_target_s,
        )
    except Exception:
        _log.exception("evals scoring failed for run %s", run_id)
        _remember_eval(run_id, None)
        return None

    _remember_eval(run_id, scores)

    try:
        entry = {
            "run_id": run_id,
            "ts": time.time(),
            "user_id": user_id,
            "turn_id": turn_id,
            **scores,
        }
        logs_dir = Path(settings.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        with open(logs_dir / "evals.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        _log.exception("evals JSONL persist failed for run %s", run_id)

    return scores


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, background_tasks: BackgroundTasks):
    guard = check_input(req.query)
    if not guard["allowed"]:
        return ChatResponse(
            user_id=req.user_id,
            query=req.query,
            response=guard["fallback"],
            candidates=[],
            affect="NEUTRAL",
            llm_tier="none",
            llm_model="none",
            retrieval_mode="none",
            latency={},
            guardrail_passed=False,
            turn_id=0,
        )

    session = _get_or_init_session(req.user_id)
    initial_state = _build_initial_state(req, session)

    result: PipelineState = run_pipeline(initial_state)

    session["session_history"] = result["session_history"]
    session["bucket_priors"] = result["bucket_priors"]
    session["type_priors"] = result["type_priors"]
    session["last_state"] = result

    affect_emotion = (result.get("affect") or {}).get("emotion", "NEUTRAL")
    run_id = result.get("run_id")

    # Evals (NLI cross-encoder) run off the response path; UI polls /evals.
    if run_id and settings.evals_enabled:
        _reserve_eval_slot(run_id)
        background_tasks.add_task(
            _compute_and_persist_evals,
            run_id=run_id,
            user_id=req.user_id,
            turn_id=result["turn_id"],
            response=result["selected_response"] or "",
            chunks=list(result.get("retrieved_chunks") or []),
            latency_log=dict(result.get("latency_log") or {}),
            affect=affect_emotion,
            gesture_tag=req.gesture_tag,
            gaze_bucket=req.gaze_bucket,
        )

    return ChatResponse(
        user_id=req.user_id,
        query=req.query,
        response=result["selected_response"] or "",
        candidates=[CandidateOut(**c) for c in result.get("candidates") or []],
        affect=affect_emotion,
        llm_tier=result.get("llm_tier_used", "unknown"),
        llm_model=result.get("llm_model_used", "unknown"),
        retrieval_mode=result.get("retrieval_mode_used", "unknown"),
        latency=result.get("latency_log") or {},
        guardrail_passed=result.get("guardrail_passed", True),
        run_id=run_id,
        turn_id=result["turn_id"],
        eval_scores=None,
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Server-Sent Events version of /chat. Runs intent + retrieval synchronously,
    then streams planner candidate tokens as they arrive. Final event carries the
    full ChatResponse-shaped payload.
    """
    guard = check_input(req.query)
    if not guard["allowed"]:
        payload = {
            "user_id": req.user_id,
            "query": req.query,
            "response": guard["fallback"],
            "candidates": [],
            "affect": "NEUTRAL",
            "llm_tier": "none",
            "llm_model": "none",
            "retrieval_mode": "none",
            "latency": {},
            "guardrail_passed": False,
            "turn_id": 0,
            "run_id": None,
            "eval_scores": None,
        }

        def _one_event():
            yield _sse({"type": "complete", "response": payload})

        return StreamingResponse(_one_event(), media_type="text/event-stream")

    session = _get_or_init_session(req.user_id)
    initial_state = _build_initial_state(req, session)

    def _gen():
        state = run_until_planner(initial_state)
        tier = choose_planner_tier(state)

        completion: dict | None = None
        for evt in planner_node._run_stream(state, tier=tier):
            if evt["type"] == "complete":
                completion = evt["planner_update"]
                break
            yield _sse(evt)

        if completion is None:
            yield _sse({"type": "error", "message": "planner produced no completion"})
            return

        state.update(completion)  # type: ignore[typeddict-item]
        state.update(feedback_node.run(state))  # type: ignore[typeddict-item]

        session["session_history"] = state["session_history"]
        session["bucket_priors"] = state["bucket_priors"]
        session["type_priors"] = state["type_priors"]
        session["last_state"] = state

        affect_emotion = (state.get("affect") or {}).get("emotion", "NEUTRAL")
        run_id = state.get("run_id")

        # Evals run off the response path; UI polls GET /evals/{run_id}.
        if run_id and settings.evals_enabled:
            _reserve_eval_slot(run_id)
            threading.Thread(
                target=_compute_and_persist_evals,
                kwargs=dict(
                    run_id=run_id,
                    user_id=req.user_id,
                    turn_id=state["turn_id"],
                    response=state["selected_response"] or "",
                    chunks=list(state.get("retrieved_chunks") or []),
                    latency_log=dict(state.get("latency_log") or {}),
                    affect=affect_emotion,
                    gesture_tag=req.gesture_tag,
                    gaze_bucket=req.gaze_bucket,
                ),
                daemon=True,
            ).start()

        final = {
            "user_id": req.user_id,
            "query": req.query,
            "response": state["selected_response"] or "",
            "candidates": [dict(c) for c in state.get("candidates") or []],
            "affect": affect_emotion,
            "llm_tier": state.get("llm_tier_used", "unknown"),
            "llm_model": state.get("llm_model_used", "unknown"),
            "retrieval_mode": state.get("retrieval_mode_used", "unknown"),
            "latency": state.get("latency_log") or {},
            "guardrail_passed": state.get("guardrail_passed", True),
            "run_id": run_id,
            "turn_id": state["turn_id"],
            "eval_scores": None,
        }
        yield _sse({"type": "complete", "response": final})

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/evals/{run_id}")
def get_evals(run_id: str):
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="invalid run_id")
    with _eval_lock:
        entry = _eval_results.get(run_id)
    if entry is None:
        return {"status": "unknown", "run_id": run_id, "eval_scores": None}
    if entry is _EVAL_FAILED:
        return {"status": "failed", "run_id": run_id, "eval_scores": None}
    if not entry:
        return {"status": "pending", "run_id": run_id, "eval_scores": None}
    return {"status": "ready", "run_id": run_id, "eval_scores": entry}


@app.post("/chat/turnaround", response_model=ChatResponse)
def chat_turnaround(req: TurnaroundRequest, background_tasks: BackgroundTasks):
    if req.user_id not in _sessions:
        raise HTTPException(status_code=404, detail="no active session")

    session = _sessions[req.user_id]
    last: PipelineState | None = session.get("last_state")
    if last is None:
        raise HTTPException(status_code=409, detail="no prior turn to rephrase")

    if req.turn_id is not None and req.turn_id != last["turn_id"]:
        raise HTTPException(status_code=409, detail="stale turn_id")

    # feedback.run will re-append (partner, aac_user) for this turn, so strip
    # both of those tail entries to avoid duplicating the partner line. The
    # rejected aac_user text is also excluded from the re-plan context this way.
    trimmed_history = list(last.get("session_history") or [])
    if trimmed_history and trimmed_history[-1].get("role") == "aac_user":
        trimmed_history.pop()
    if trimmed_history and trimmed_history[-1].get("role") == "partner":
        trimmed_history.pop()

    intent_kind = classify_intent_kind(last.get("intent_route"))

    gen_cfg = dict(last.get("generation_config") or {})
    if intent_kind == "present_state":
        gen_cfg["persona_mod"] = "present_state_retry"
        gen_cfg["tone_tag"] = "[TONE:HONEST_UNCERTAIN]"
    else:
        gen_cfg["persona_mod"] = "reverse_stance"
        gen_cfg.setdefault("tone_tag", "[TONE:CLARIFYING_REPHRASE]")

    replan_state: PipelineState = dict(last)  # type: ignore[assignment]
    replan_state["session_history"] = trimmed_history
    replan_state["generation_config"] = gen_cfg
    replan_state["head_signal"] = req.head_signal or last.get("head_signal")
    replan_state["turnaround_triggered"] = True
    replan_state["latency_log"] = {
        "t_sensing": 0.0,
        "t_intent": 0.0,
        "t_retrieval": 0.0,
        "t_generation": 0.0,
        "t_total": 0.0,
    }

    # For PERSONAL turnarounds, pull fresh chunks excluding the bucket and
    # exact texts of the rejected response — same chunks would just produce
    # the same wrong answer. _re_retrieve_excluding returns None when the
    # fresh batch is no better than what we already had, in which case we
    # keep the original chunks rather than degrade to lower-relevance ones.
    if intent_kind == "memory":
        fresh_chunks = _re_retrieve_excluding(
            query=last["raw_query"],
            user_id=last["user_id"],
            rejected_chunks=last.get("retrieved_chunks") or [],
        )
        if fresh_chunks is not None:
            replan_state["retrieved_chunks"] = fresh_chunks
            replan_state["retrieval_mode_used"] = "turnaround_rebucket"

    planner_update = planner_node.run_primary(replan_state)
    replan_state.update(planner_update)  # type: ignore[typeddict-item]

    feedback_update = feedback_node.run(replan_state)
    replan_state.update(feedback_update)  # type: ignore[typeddict-item]

    session["session_history"] = replan_state["session_history"]
    session["bucket_priors"] = replan_state["bucket_priors"]
    session["type_priors"] = replan_state["type_priors"]
    session["last_state"] = replan_state

    affect_emotion = (replan_state.get("affect") or {}).get("emotion", "NEUTRAL")
    run_id = replan_state.get("run_id")

    if run_id and settings.evals_enabled:
        _reserve_eval_slot(run_id)
        background_tasks.add_task(
            _compute_and_persist_evals,
            run_id=run_id,
            user_id=req.user_id,
            turn_id=replan_state["turn_id"],
            response=replan_state["selected_response"] or "",
            chunks=list(replan_state.get("retrieved_chunks") or []),
            latency_log=dict(replan_state.get("latency_log") or {}),
            affect=affect_emotion,
            gesture_tag=replan_state.get("gesture_tag"),
            gaze_bucket=replan_state.get("gaze_bucket"),
        )

    return ChatResponse(
        user_id=req.user_id,
        query=replan_state["raw_query"],
        response=replan_state["selected_response"] or "",
        candidates=[CandidateOut(**c) for c in replan_state.get("candidates") or []],
        affect=affect_emotion,
        llm_tier=replan_state.get("llm_tier_used", "unknown"),
        llm_model=replan_state.get("llm_model_used", "unknown"),
        retrieval_mode=replan_state.get("retrieval_mode_used", "unknown"),
        latency=replan_state.get("latency_log") or {},
        guardrail_passed=replan_state.get("guardrail_passed", True),
        run_id=run_id,
        turn_id=replan_state["turn_id"],
        eval_scores=None,
    )


def _find_turn_from_jsonl(run_id: str) -> dict | None:
    """Scan turns.jsonl from the end for a matching run_id. Used as fallback
    when the session's last_state has already moved on."""
    path = Path(settings.logs_dir) / "turns.jsonl"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None
    for line in reversed(lines[-500:]):  # bounded tail scan
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("run_id") == run_id:
            return row
    return None


@app.post("/chat/pick")
def pick_candidate(req: PickRequest):
    if not _RUN_ID_RE.match(req.run_id):
        raise HTTPException(status_code=400, detail="invalid run_id")

    session = _sessions.get(req.user_id) or {}
    last = session.get("last_state") or {}
    candidates = last.get("candidates") or []
    query_text = last.get("raw_query") or ""

    # Fallback: last_state already advanced past this run_id — read from JSONL
    if last.get("run_id") != req.run_id or not candidates:
        row = _find_turn_from_jsonl(req.run_id)
        if not row:
            raise HTTPException(status_code=404, detail="turn not found")
        candidates = row.get("candidates") or []
        query_text = row.get("query") or query_text

    if req.picked_idx >= len(candidates):
        raise HTTPException(status_code=400, detail="picked_idx out of range")

    picked = candidates[req.picked_idx]
    picked_text = picked.get("text", "")
    strategy = picked.get("strategy", "unknown")
    picked_buckets = [
        b for b in (picked.get("grounded_buckets") or []) if b and b != "open_domain"
    ]

    if query_text and picked_text:
        try:
            pick_index.add(
                query=query_text,
                user_id=req.user_id,
                strategy=strategy,
                picked_text=picked_text,
                picked_buckets=picked_buckets,
            )
        except Exception as exc:
            _log.warning("pick_index add failed: %r", exc)

    logs_dir = Path(settings.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "run_id": req.run_id,
        "user_id": req.user_id,
        "picked_idx": req.picked_idx,
        "strategy": strategy,
        "picked_text": picked_text,
        "query": query_text,
    }
    with open(logs_dir / "picks.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {"status": "ok", "strategy": strategy}


@app.post("/chat/regenerate/stream")
def chat_regenerate_stream(req: RegenerateRequest):
    """Streaming regenerate — same as /chat/stream but reuses last_state and
    marks all prior candidates as rejected."""
    if req.user_id not in _sessions:
        raise HTTPException(status_code=404, detail="no active session")
    session = _sessions[req.user_id]
    last: PipelineState | None = session.get("last_state")
    if last is None:
        raise HTTPException(status_code=409, detail="no prior turn to regenerate")
    if req.turn_id is not None and req.turn_id != last["turn_id"]:
        raise HTTPException(status_code=409, detail="stale turn_id")

    gen_cfg = dict(last.get("generation_config") or {})
    gen_cfg["persona_mod"] = "all_rejected"
    gen_cfg.setdefault("tone_tag", "[TONE:TRY_DIFFERENT_ANGLE]")

    prior_rejected = [c.get("text", "") for c in (last.get("candidates") or [])]
    merged = (
        list(last.get("rejected_candidates") or [])
        + [t for t in prior_rejected if t]
        + [t for t in req.rejected_texts if t]
    )
    seen: set[str] = set()
    rejected: list[str] = []
    for t in merged:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            rejected.append(t)

    trimmed_history = list(last.get("session_history") or [])
    if trimmed_history and trimmed_history[-1].get("role") == "aac_user":
        trimmed_history.pop()
    if trimmed_history and trimmed_history[-1].get("role") == "partner":
        trimmed_history.pop()

    replan_state: PipelineState = dict(last)  # type: ignore[assignment]
    replan_state["session_history"] = trimmed_history
    replan_state["generation_config"] = gen_cfg
    replan_state["rejected_candidates"] = rejected
    replan_state["turnaround_triggered"] = False
    replan_state["latency_log"] = {
        "t_sensing": 0.0,
        "t_intent": 0.0,
        "t_retrieval": 0.0,
        "t_generation": 0.0,
        "t_total": 0.0,
    }

    def _gen():
        completion: dict | None = None
        for evt in planner_node._run_stream(replan_state, tier="primary"):
            if evt["type"] == "complete":
                completion = evt["planner_update"]
                break
            yield _sse(evt)
        if completion is None:
            yield _sse({"type": "error", "message": "planner produced no completion"})
            return
        replan_state.update(completion)  # type: ignore[typeddict-item]
        replan_state.update(feedback_node.run(replan_state))  # type: ignore[typeddict-item]

        session["session_history"] = replan_state["session_history"]
        session["bucket_priors"] = replan_state["bucket_priors"]
        session["type_priors"] = replan_state["type_priors"]
        session["last_state"] = replan_state

        affect_emotion = (replan_state.get("affect") or {}).get("emotion", "NEUTRAL")
        run_id = replan_state.get("run_id")
        eval_scores = _compute_and_persist_evals(
            run_id=run_id,
            user_id=req.user_id,
            turn_id=replan_state["turn_id"],
            response=replan_state["selected_response"] or "",
            chunks=list(replan_state.get("retrieved_chunks") or []),
            latency_log=dict(replan_state.get("latency_log") or {}),
            affect=affect_emotion,
            gesture_tag=replan_state.get("gesture_tag"),
            gaze_bucket=replan_state.get("gaze_bucket"),
        )
        final = {
            "user_id": req.user_id,
            "query": replan_state["raw_query"],
            "response": replan_state["selected_response"] or "",
            "candidates": [dict(c) for c in replan_state.get("candidates") or []],
            "affect": affect_emotion,
            "llm_tier": replan_state.get("llm_tier_used", "unknown"),
            "llm_model": replan_state.get("llm_model_used", "unknown"),
            "retrieval_mode": replan_state.get("retrieval_mode_used", "unknown"),
            "latency": replan_state.get("latency_log") or {},
            "guardrail_passed": replan_state.get("guardrail_passed", True),
            "run_id": run_id,
            "turn_id": replan_state["turn_id"],
            "eval_scores": eval_scores,
        }
        yield _sse({"type": "complete", "response": final})

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/regenerate", response_model=ChatResponse)
def chat_regenerate(req: RegenerateRequest):
    """Re-run the planner for the same turn with all prior candidates marked rejected.
    Does NOT advance turn_id — same partner query, fresh fan-out of candidates.
    """
    if req.user_id not in _sessions:
        raise HTTPException(status_code=404, detail="no active session")
    session = _sessions[req.user_id]
    last: PipelineState | None = session.get("last_state")
    if last is None:
        raise HTTPException(status_code=409, detail="no prior turn to regenerate")
    if req.turn_id is not None and req.turn_id != last["turn_id"]:
        raise HTTPException(status_code=409, detail="stale turn_id")

    gen_cfg = dict(last.get("generation_config") or {})
    gen_cfg["persona_mod"] = "all_rejected"
    gen_cfg.setdefault("tone_tag", "[TONE:TRY_DIFFERENT_ANGLE]")

    prior_rejected = [c.get("text", "") for c in (last.get("candidates") or [])]
    merged_rejected = (
        list(last.get("rejected_candidates") or [])
        + [t for t in prior_rejected if t]
        + [t for t in req.rejected_texts if t]
    )
    # Dedupe while preserving order.
    seen: set[str] = set()
    rejected: list[str] = []
    for t in merged_rejected:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            rejected.append(t)

    # Strip the tail (partner, aac_user) so feedback doesn't stack duplicate
    # history entries on every regenerate — the user hasn't committed yet.
    trimmed_history = list(last.get("session_history") or [])
    if trimmed_history and trimmed_history[-1].get("role") == "aac_user":
        trimmed_history.pop()
    if trimmed_history and trimmed_history[-1].get("role") == "partner":
        trimmed_history.pop()

    replan_state: PipelineState = dict(last)  # type: ignore[assignment]
    replan_state["session_history"] = trimmed_history
    replan_state["generation_config"] = gen_cfg
    replan_state["rejected_candidates"] = rejected
    replan_state["turnaround_triggered"] = False  # keep multi-shot
    replan_state["latency_log"] = {
        "t_sensing": 0.0,
        "t_intent": 0.0,
        "t_retrieval": 0.0,
        "t_generation": 0.0,
        "t_total": 0.0,
    }

    planner_update = planner_node.run_primary(replan_state)
    replan_state.update(planner_update)  # type: ignore[typeddict-item]

    # Feedback node rewrites history + assigns a new run_id. Each regenerate
    # is its own row in turns.jsonl for the eval record.
    feedback_update = feedback_node.run(replan_state)
    replan_state.update(feedback_update)  # type: ignore[typeddict-item]

    session["session_history"] = replan_state["session_history"]
    session["bucket_priors"] = replan_state["bucket_priors"]
    session["type_priors"] = replan_state["type_priors"]
    session["last_state"] = replan_state

    affect_emotion = (replan_state.get("affect") or {}).get("emotion", "NEUTRAL")
    run_id = replan_state.get("run_id")

    eval_scores = _compute_and_persist_evals(
        run_id=run_id,
        user_id=req.user_id,
        turn_id=replan_state["turn_id"],
        response=replan_state["selected_response"] or "",
        chunks=list(replan_state.get("retrieved_chunks") or []),
        latency_log=dict(replan_state.get("latency_log") or {}),
        affect=affect_emotion,
        gesture_tag=replan_state.get("gesture_tag"),
        gaze_bucket=replan_state.get("gaze_bucket"),
    )

    return ChatResponse(
        user_id=req.user_id,
        query=replan_state["raw_query"],
        response=replan_state["selected_response"] or "",
        candidates=[CandidateOut(**c) for c in replan_state.get("candidates") or []],
        affect=affect_emotion,
        llm_tier=replan_state.get("llm_tier_used", "unknown"),
        llm_model=replan_state.get("llm_model_used", "unknown"),
        retrieval_mode=replan_state.get("retrieval_mode_used", "unknown"),
        latency=replan_state.get("latency_log") or {},
        guardrail_passed=replan_state.get("guardrail_passed", True),
        run_id=run_id,
        turn_id=replan_state["turn_id"],
        eval_scores=eval_scores,
    )


@app.post("/feedback/rating")
def submit_rating(req: RatingRequest):
    if not _RUN_ID_RE.match(req.run_id):
        raise HTTPException(status_code=400, detail="invalid run_id")
    logs_dir = Path(settings.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "run_id": req.run_id,
        "user_id": req.user_id,
        "authenticity": req.authenticity,
        "rater_id": req.rater_id,
        "notes": req.notes,
    }
    with open(logs_dir / "ratings.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"status": "ok"}


class InkRecognizeRequest(BaseModel):
    image_base64: str


@lru_cache(maxsize=1)
def _get_vision_client():
    from openai import OpenAI as _OpenAI

    return _OpenAI(
        base_url=settings.ink_vision_base_url,
        api_key=settings.ink_vision_api_key or "unused",
    )


@app.post("/ink/recognize")
def ink_recognize(req: InkRecognizeRequest):
    if not req.image_base64:
        return {"text": ""}
    if not settings.ink_vision_api_key:
        _log.warning("/ink/recognize called but INK_VISION_API_KEY is not set")
        raise HTTPException(status_code=503, detail="INK_VISION_API_KEY not configured")
    try:
        client = _get_vision_client()
        response = client.chat.completions.create(
            model=settings.ink_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{req.image_base64}"
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "This is a single handwritten character or short word "
                                "drawn in the air. Reply with ONLY the character or "
                                "word, nothing else."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=64,
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        _log.info("/ink/recognize raw → %r", raw[:200])
        # Strip <think>…</think> blocks emitted by reasoning models, harmless on others.
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        _log.info("/ink/recognize → %r", text)
        return {"text": text}
    except Exception as exc:
        _log.exception("/ink/recognize failed: %r", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Serve React frontend — must be last so API routes take priority
_frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
