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
    messages = _build_messages(
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
        messages=messages,
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

    augmented_prompt = "\n\n".join(m["content"] for m in messages)
    return {
        "augmented_prompt": augmented_prompt,
        "candidates": [selected],
        "selected_response": selected,
        "llm_tier_used": tier,
        "llm_model_used": active_model(tier),
        "latency_log": latency_log,
        "guardrail_passed": guard["passed"],
    }


def _resolve_tone_tag(user_id: str, affect: str, default_tag: str) -> str:
    return _PERSONA_TONE_OVERRIDES.get(user_id, {}).get(affect, default_tag)


def _build_messages(
    profile: dict,
    chunks: list[dict],
    history: list[dict],
    query: str,
    tone_tag: str,
    gen_cfg: dict,
    gesture_tag: str | None = None,
    air_written_text: str | None = None,
) -> list[dict]:
    # Split into a stable system message (same per persona — gets cached by the
    # provider) and a turn-specific user message. Anything that changes per
    # turn (retrieval, affect, gesture, partner query) must live in the user
    # message or the prefix cache invalidates.
    system_content = _build_system(profile)
    user_content = _build_user(
        chunks,
        history,
        query,
        tone_tag,
        gen_cfg,
        gesture_tag,
        air_written_text,
        profile["name"],
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _build_system(profile: dict) -> str:
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

    return f"""\
You are {profile["name"]}. Reply in first person as them, in 1–3 sentences. \
Never narrate, analyze, describe, or list traits about your character. \
Never say "As an AI", "The user wants me to", "Key characteristics", or anything meta. \
Just speak.

--- Character sheet (reference only — do NOT quote or paraphrase this block) ---
Condition: {profile["condition"]}
Access: {access}
Voice: {style_summary}

Style examples (match this register when you speak):
  {style_exemplar}

Answering rules:
- For personal questions: use ONLY the memories in the user message; if they don't cover it, say "I don't know."
- For general-knowledge questions: answer from what you know, in your voice.
- Keep it to 1–3 sentences, first person, no meta-commentary.
--- end character sheet ---"""


_PERSONA_MOD_INSTRUCTIONS = {
    "amplify_quirks": "Amplify your characteristic style and personality.",
    "suppress_humor": "Be direct and supportive. Suppress humor.",
    "baseline": "Use your natural communication style.",
    "add_confirmation": "Add a clarifying question or confirmation at the end.",
}


def _build_user(
    chunks: list[dict],
    history: list[dict],
    query: str,
    tone_tag: str,
    gen_cfg: dict,
    gesture_tag: str | None,
    air_written_text: str | None,
    persona_name: str,
) -> str:
    personal_chunks = [c for c in chunks if c.get("source", "personal") == "personal"]
    contextual_chunks = [c for c in chunks if c.get("source") == "contextual"]
    open_domain_chunks = [c for c in chunks if c.get("source") == "open_domain"]

    memory_block = (
        "\n".join(
            f"  [{c['bucket']}/{c.get('type', 'narrative')}] {c['text']}"
            for c in personal_chunks
        )
        or "  (no memories retrieved)"
    )
    contextual_block = (
        "\n".join(f"  {c['text']}" for c in contextual_chunks)
        or "  (nothing relevant from this session)"
    )
    open_domain_note = (
        "  Treat this sub-query as general knowledge; answer from what you know.\n"
        + "\n".join(f"  {c['text']}" for c in open_domain_chunks)
        if open_domain_chunks
        else "  (none)"
    )
    history_block = (
        "\n".join(f"  {h.get('role', '?')}: {h.get('content', '')}" for h in history)
        or "  (start of session)"
    )

    gesture_line = ""
    if gesture_tag:
        g_tag = GESTURE_TO_TAG.get(gesture_tag, f"[GESTURE:{gesture_tag}]")
        gesture_line = f"\nActive gesture signal: {g_tag}"

    air_writing_line = ""
    if air_written_text:
        air_writing_line = f'\nThe user air-wrote: "{air_written_text}" — treat as supplementary intent.'

    persona_instruction = _PERSONA_MOD_INSTRUCTIONS.get(
        gen_cfg.get("persona_mod", "baseline"),
        _PERSONA_MOD_INSTRUCTIONS["baseline"],
    )

    return f"""\
{tone_tag}{gesture_line}{air_writing_line}
{persona_instruction}

Personal memories:
{memory_block}

From earlier in this conversation:
{contextual_block}

General knowledge note:
{open_domain_note}

Recent conversation:
{history_block}

Partner just said: {query}

Your reply as {persona_name} (1–3 sentences, first person):"""
