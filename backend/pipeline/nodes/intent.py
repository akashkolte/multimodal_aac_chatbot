# Intent decomposition node — LLM-based query classification and routing.
from __future__ import annotations

import re
import time
from typing import Literal

from pydantic import BaseModel

from backend.config.settings import settings
from backend.generation.llm_client import chat_complete
from backend.pipeline.state import GenerationConfig, IntentRoute, PipelineState

# ── Pydantic output schemas ────────────────────────────────────────────────────

BucketType = Literal["family", "medical", "hobbies", "daily_routine", "social"]
AffectEmotion = Literal["HAPPY", "FRUSTRATED", "NEUTRAL", "SURPRISED"]


class SubIntentSchema(BaseModel):
    type: Literal["PERSONAL", "CONTEXTUAL", "OPEN_DOMAIN"]
    query: str
    bucket_hint: BucketType | None = None
    priority: Literal["fast", "normal"] = "normal"


class StyleConfig(BaseModel):
    tone_tag: str  # e.g. "[TONE:WITTY_SARCASTIC]"
    max_tokens: int
    retrieval_mode: str  # "fast" | "full"
    persona_mod: (
        str  # "amplify_quirks" | "suppress_humor" | "baseline" | "add_confirmation"
    )


class IntentRouteSchema(BaseModel):
    sub_intents: list[SubIntentSchema]
    style_constraints: StyleConfig
    affect: AffectEmotion


# ── Affect → generation config mapping ────────────────────────────────────────

_AFFECT_CONFIG: dict[str, GenerationConfig] = {
    "HAPPY": {
        "max_tokens": settings.max_tokens_happy,
        "tone_tag": "[TONE:WARM]",
        "retrieval_mode": "full",
        "persona_mod": "amplify_quirks",
    },
    "FRUSTRATED": {
        "max_tokens": settings.max_tokens_frustrated,
        "tone_tag": "[TONE:DIRECT_EMPATHETIC]",
        "retrieval_mode": "fast",
        "persona_mod": "suppress_humor",
    },
    "NEUTRAL": {
        "max_tokens": settings.max_tokens_neutral,
        "tone_tag": "[TONE:DEFAULT]",
        "retrieval_mode": "full",
        "persona_mod": "baseline",
    },
    "SURPRISED": {
        "max_tokens": settings.max_tokens_surprised,
        "tone_tag": "[TONE:CLARIFYING]",
        "retrieval_mode": "full",
        "persona_mod": "add_confirmation",
    },
}

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the intent decomposition controller for an AAC (Augmentative and \
Alternative Communication) chatbot. Given a partner's query and the AAC \
user's current affect state, classify each intent and produce routing \
instructions in the required JSON format.

Intent types:
- PERSONAL: requires autobiographical memory retrieval
- CONTEXTUAL: answerable from session history
- OPEN_DOMAIN: answerable from general knowledge (no retrieval needed)

Bucket hints (only for PERSONAL): family | medical | hobbies | daily_routine | social
Priority: set "fast" when affect is FRUSTRATED to reduce latency.

Respond ONLY with valid JSON matching the IntentRoute schema. No extra text.
"""


def _build_user_prompt(
    query: str, affect: str, persona_name: str, air_written_text: str | None = None
) -> str:
    air_note = (
        f'\nAir-written supplement: "{air_written_text}"' if air_written_text else ""
    )
    return (
        f"Persona: {persona_name}\n"
        f"Affect: {affect}\n"
        f"Partner query: {query}{air_note}\n\n"
        "Produce the IntentRoute JSON:"
    )


# ── Node entry point ───────────────────────────────────────────────────────────


def run(state: PipelineState) -> dict:
    """LangGraph node: intent decomposition."""
    t0 = time.perf_counter()

    # --fast mode: intent_route already resolved by keyword routing in main.py
    if state.get("intent_route") and state.get("generation_config"):
        return {}  # nothing to update — downstream nodes use the pre-filled values

    affect_state = state.get("affect") or {}
    emotion: str = affect_state.get("emotion", "NEUTRAL")
    query: str = state["raw_query"]
    persona_name: str = state["persona_profile"].get("name", "unknown")

    gen_config = _AFFECT_CONFIG.get(emotion, _AFFECT_CONFIG["NEUTRAL"])

    route: IntentRoute | None = None
    last_error: str = ""

    for attempt in range(3):  # LangGraph retry logic (up to 2 retries)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(
                    query,
                    emotion,
                    persona_name,
                    air_written_text=state.get("air_written_text"),
                ),
            },
        ]
        if attempt > 0:
            messages.append(
                {
                    "role": "user",
                    "content": f"Validation error: {last_error}. Fix and retry.",
                }
            )

        raw = chat_complete(
            messages=messages,
            max_tokens=512,
            temperature=0.0,
        )

        try:
            # Strip markdown fences (```json ... ```) that many models add
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned.strip())
            parsed = IntentRouteSchema.model_validate_json(cleaned)
            route = {
                "sub_intents": [si.model_dump() for si in parsed.sub_intents],
                "style_constraints": parsed.style_constraints.model_dump(),
                "affect": parsed.affect,
            }
            break
        except Exception as exc:
            last_error = str(exc)

    if route is None:
        # Hard fallback: treat as a single PERSONAL intent, full retrieval
        route = {
            "sub_intents": [
                {
                    "type": "PERSONAL",
                    "query": query,
                    "bucket_hint": None,
                    "priority": "normal",
                }
            ],
            "style_constraints": gen_config,
            "affect": emotion,
        }

    t_intent = time.perf_counter() - t0

    latency_log = dict(state.get("latency_log") or {})
    latency_log["t_intent"] = round(t_intent, 4)

    return {
        "intent_route": route,
        "generation_config": gen_config,
        "latency_log": latency_log,
    }
