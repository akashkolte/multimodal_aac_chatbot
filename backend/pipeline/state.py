# Typed state flowing through every pipeline node.
from __future__ import annotations

from typing import Any

from typing_extensions import TypedDict

# ── Sub-types ──────────────────────────────────────────────────────────────────


class AffectVector(TypedDict):
    MAR: float  # Mouth Aspect Ratio
    EAR: float  # Eye Aspect Ratio
    BRI: float  # Brow Raise Index
    LCP: float  # Lip Corner Pull


class AffectState(TypedDict):
    emotion: str  # "HAPPY" | "FRUSTRATED" | "NEUTRAL" | "SURPRISED"
    vector: AffectVector
    smoothed: AffectVector  # EMA-smoothed vector


class RetrievedChunk(TypedDict):
    text: str
    bucket: str  # family | medical | hobbies | daily_routine | social | contextual | open_domain
    type: str  # narrative | social_post | chat_log  (personal chunks only)
    user: str
    score: float  # cosine similarity from the embedder
    source: str  # "personal" | "contextual" | "open_domain"


class SubIntent(TypedDict):
    type: str  # "PERSONAL" | "CONTEXTUAL" | "PRESENT_STATE" | "OPEN_DOMAIN"
    query: str
    bucket_hint: str | None
    priority: str  # "fast" | "normal"


class IntentRoute(TypedDict):
    sub_intents: list[SubIntent]
    style_constraints: dict[str, Any]  # tone, max_tokens, etc.
    affect: str


class StyleDirective(TypedDict):
    tone_tag: str  # e.g. "[TONE:WARM]" — kept for logging + eval
    register: str  # short register phrase, e.g. "warm, upbeat, affectionate"
    prefer_words: list[str]  # lexical bias — words to steer toward
    avoid_words: list[str]  # anti-patterns — words to steer away from
    opener_hint: str | None  # structural hint for the opening clause
    exemplar: str  # one short sentence in the target register


class GenerationConfig(TypedDict):
    max_tokens: int
    tone_tag: str  # legacy tag (kept in sync with style["tone_tag"] for existing log consumers)
    retrieval_mode: str  # "fast" | "full"
    persona_mod: str
    # persona_mod values:
    #   "amplify_quirks" | "suppress_humor" | "baseline"
    #   | "add_confirmation" | "turnaround"
    #   | "reverse_stance" | "present_state_retry"
    style: StyleDirective


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
    persona_profile: dict[str, Any]  # full profile from users.json
    session_history: list[dict]
    turn_id: int

    # ── L1: Sensing outputs ───────────────────────────────────────────────────
    affect: AffectState | None
    gesture_tag: str | None  # e.g. "THUMBS_UP"
    gaze_bucket: str | None  # bucket hinted by gaze fixation
    air_written_text: str | None  # concatenated air-written chars
    head_signal: str | None  # "HEAD_SHAKE" | "HEAD_NOD_DISSATISFIED"
    turnaround_triggered: bool  # true when re-planned from dissatisfaction signal

    # ── L2: Intent decomposition outputs ─────────────────────────────────────
    raw_query: str  # partner's typed/spoken query
    intent_route: IntentRoute | None  # Pydantic-validated routing
    generation_config: GenerationConfig | None

    # ── L3: Retrieval outputs ─────────────────────────────────────────────────
    retrieved_chunks: list[RetrievedChunk]
    bucket_priors: dict[str, float]  # session-level Bayesian priors
    retrieval_mode_used: str  # "fast" | "full"

    # ── L4: Generation outputs ────────────────────────────────────────────────
    augmented_prompt: str | None
    candidates: list[str]  # 2-3 candidate responses
    selected_response: str | None
    llm_tier_used: str  # "primary" | "fallback"
    llm_model_used: str  # actual model name (e.g. "gemma4:31b-cloud")

    # ── L5: Feedback / tracking ───────────────────────────────────────────────
    latency_log: LatencyLog | None
    run_id: str | None  # UUID assigned per turn; logged to logs/turns.jsonl
    guardrail_passed: bool
