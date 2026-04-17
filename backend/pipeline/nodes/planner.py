# Planner node — prompt building, candidate generation, composite ranking.
from __future__ import annotations

import time

from backend.config.settings import settings
from backend.generation.llm_client import active_model, chat_complete
from backend.guardrails.checks import check_output
from backend.pipeline.state import PipelineState
from backend.sensing.labels import GESTURE_TO_TAG

# ── Persona-specific tone tags (applied on top of affect base tag) ─────────────

_PERSONA_TONE_OVERRIDES: dict[str, dict[str, str]] = {
    "mia_chen": {
        "HAPPY": "[TONE:WITTY_SARCASTIC]",
        "FRUSTRATED": "[TONE:DIRECT_EMPATHETIC]",
    },
    "gerald_okafor": {
        "HAPPY": "[TONE:WARM_FORMAL]",
        "FRUSTRATED": "[TONE:MEASURED_EMPATHETIC]",
    },
    "arjun_mehta": {
        "HAPPY": "[TONE:DIRECT_WARM]",
        "FRUSTRATED": "[TONE:MINIMAL_DIRECT]",
    },
}


def run_primary(state: PipelineState) -> dict:
    return _run(state, tier="primary")


def run_fallback(state: PipelineState) -> dict:
    return _run(state, tier="fallback")


# ── Core implementation ────────────────────────────────────────────────────────


def _run(state: PipelineState, tier: str) -> dict:
    t0 = time.perf_counter()

    profile = state["persona_profile"]
    user_id = state["user_id"]
    affect = (state.get("affect") or {}).get("emotion", "NEUTRAL")
    gen_cfg = state.get("generation_config") or {}
    chunks = state.get("retrieved_chunks") or []
    history = (state.get("session_history") or [])[-20:]

    tone_tag = _resolve_tone_tag(
        user_id, affect, gen_cfg.get("tone_tag", "[TONE:DEFAULT]")
    )
    gesture_tag = state.get("gesture_tag")
    air_written_text = state.get("air_written_text")
    prompt = _build_prompt(
        profile,
        chunks,
        history,
        state["raw_query"],
        tone_tag,
        gen_cfg,
        gesture_tag=gesture_tag,
        air_written_text=air_written_text,
    )

    selected = chat_complete(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=gen_cfg.get("max_tokens", settings.max_tokens_neutral),
        temperature=0.4,
        tier=tier,
    )

    guard = check_output(selected, chunks)
    if not guard["passed"]:
        selected = guard["fallback"]

    t_gen = time.perf_counter() - t0
    latency_log = dict(state.get("latency_log") or {})
    latency_log["t_generation"] = round(t_gen, 4)
    latency_log["t_total"] = round(
        latency_log.get("t_sensing", 0)
        + latency_log.get("t_intent", 0)
        + latency_log.get("t_retrieval", 0)
        + t_gen,
        4,
    )

    return {
        "augmented_prompt": prompt,
        "candidates": [selected],
        "selected_response": selected,
        "llm_tier_used": tier,
        "llm_model_used": active_model(tier),
        "latency_log": latency_log,
        "guardrail_passed": guard["passed"],
    }


def _resolve_tone_tag(user_id: str, affect: str, default_tag: str) -> str:
    return _PERSONA_TONE_OVERRIDES.get(user_id, {}).get(affect, default_tag)


def _build_prompt(
    profile: dict,
    chunks: list[dict],
    history: list[dict],
    query: str,
    tone_tag: str,
    gen_cfg: dict,
    gesture_tag: str | None = None,
    air_written_text: str | None = None,
) -> str:
    memory_block = (
        "\n".join(
            f"  [{c['bucket']}/{c.get('type', 'narrative')}] {c['text']}"
            for c in chunks
        )
        or "  (no memories retrieved)"
    )
    history_block = (
        "\n".join(f"  {h.get('role', '?')}: {h.get('content', '')}" for h in history)
        or "  (start of session)"
    )

    prefs = profile.get("stylistic_preferences") or {}
    style_bits = []
    if prefs.get("tone"):
        style_bits.append("tone: " + ", ".join(prefs["tone"]))
    if prefs.get("humor"):
        style_bits.append("humor: " + prefs["humor"])
    if prefs.get("sentence_length"):
        style_bits.append("sentence length: " + prefs["sentence_length"])
    if prefs.get("formality"):
        style_bits.append("formality: " + prefs["formality"])
    style_summary = (
        "; ".join(style_bits) or profile.get("style") or "natural, conversational"
    )

    exemplars = prefs.get("example_phrases") or []
    style_exemplar = "\n  ".join(exemplars) if exemplars else "(no exemplar)"

    access = (profile.get("access_needs") or {}).get("input_method") or "an AAC device"

    gesture_line = ""
    if gesture_tag:
        g_tag = GESTURE_TO_TAG.get(gesture_tag, f"[GESTURE:{gesture_tag}]")
        gesture_line = f"\nActive gesture signal: {g_tag}"

    air_writing_line = ""
    if air_written_text:
        air_writing_line = f'\nThe user air-wrote: "{air_written_text}" — treat as supplementary intent.'

    persona_mod = gen_cfg.get("persona_mod", "baseline")
    persona_instruction = {
        "amplify_quirks": "Amplify your characteristic style and personality.",
        "suppress_humor": "Be direct and supportive. Suppress humor.",
        "baseline": "Use your natural communication style.",
        "add_confirmation": "Add a clarifying question or confirmation at the end.",
    }.get(persona_mod, "Use your natural communication style.")

    return f"""\
You are {profile["name"]}. You have {profile["condition"]} and communicate through {access}, but your voice and thoughts are fully your own.
Communication style: {style_summary}
{tone_tag}{gesture_line}{air_writing_line}

Style exemplars — match this register:
  {style_exemplar}

Personal memories (use ONLY these for personal facts; each tagged [bucket/type] where type is narrative, social_post, or chat_log):
{memory_block}

Recent conversation:
{history_block}

Partner says: {query}

Instructions:
- Speak in first person as {profile["name"]}.
- {persona_instruction}
- Keep response to 1-3 sentences.
- If the answer isn't in your memories, say "I don't know."
- Do NOT say "As an AI" or break persona.

Response:"""
