"""
Typed state object that flows through every LangGraph node.

Each node receives the full PipelineState and returns a dict
containing only the keys it updates — LangGraph merges them.
"""
from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
import operator


# ── Sub-types ──────────────────────────────────────────────────────────────────

class AffectVector(TypedDict):
    MAR: float   # Mouth Aspect Ratio
    EAR: float   # Eye Aspect Ratio
    BRI: float   # Brow Raise Index
    LCP: float   # Lip Corner Pull


class AffectState(TypedDict):
    emotion: str          # "HAPPY" | "FRUSTRATED" | "NEUTRAL" | "SURPRISED"
    vector: AffectVector
    smoothed: AffectVector  # EMA-smoothed vector


class RetrievedChunk(TypedDict):
    text: str
    bucket: str           # family | medical | hobbies | daily_routine | social
    user: str
    score: float          # cross-encoder rerank score


class SubIntent(TypedDict):
    type: str             # "PERSONAL" | "CONTEXTUAL" | "OPEN_DOMAIN"
    query: str
    bucket_hint: Optional[str]
    priority: str         # "fast" | "normal"


class IntentRoute(TypedDict):
    sub_intents: list[SubIntent]
    style_constraints: dict[str, Any]   # tone, max_tokens, etc.
    affect: str


class GenerationConfig(TypedDict):
    max_tokens: int
    tone_tag: str         # e.g. "[TONE:WITTY_SARCASTIC]"
    retrieval_mode: str   # "fast" | "full"
    persona_mod: str      # "amplify_quirks" | "suppress_humor" | "baseline" | "add_confirmation"


class LatencyLog(TypedDict):
    t_sensing: float
    t_intent: float
    t_retrieval: float
    t_generation: float
    t_total: float


# ── Main pipeline state ────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # ── Session context (set at turn start, stable across nodes) ──────────────
    user_id: str
    persona_profile: dict[str, Any]          # full profile from users.json
    session_history: Annotated[list[dict], operator.add]  # auto-appended
    turn_id: int

    # ── L1: Sensing outputs ───────────────────────────────────────────────────
    affect: Optional[AffectState]
    gesture_tag: Optional[str]               # e.g. "THUMBS_UP"
    gaze_bucket: Optional[str]               # bucket hinted by gaze fixation
    air_written_text: Optional[str]          # concatenated air-written chars

    # ── L2: Intent decomposition outputs ─────────────────────────────────────
    raw_query: str                           # partner's typed/spoken query
    intent_route: Optional[IntentRoute]      # Pydantic-validated routing
    generation_config: Optional[GenerationConfig]

    # ── L3: Retrieval outputs ─────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]
    bucket_priors: dict[str, float]          # session-level Bayesian priors
    retrieval_mode_used: str                 # "fast" | "full"

    # ── L4: Generation outputs ────────────────────────────────────────────────
    augmented_prompt: Optional[str]
    candidates: list[str]                    # 2-3 candidate responses
    selected_response: Optional[str]
    llm_tier_used: str                       # "primary" | "fallback" | "local"

    # ── L5: Feedback / tracking ───────────────────────────────────────────────
    latency_log: Optional[LatencyLog]
    mlflow_run_id: Optional[str]
    guardrail_passed: bool
