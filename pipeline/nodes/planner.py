"""
L4 — Dialogue Planning & Generation node.

Expression-conditioned response shaping (proposal §5.5):
  1. Build augmented prompt (persona profile + retrieved evidence + affect config + style exemplar)
  2. Generate N candidate responses
  3. Rank candidates by composite score: α·faithful + β·style + γ·affect_match
  4. Return the top-ranked response

Two entry points:
  run_primary  — Qwen3-30B-A3B (or configured primary tier)
  run_fallback — Qwen3-8B (faster, triggered by latency threshold)
"""
from __future__ import annotations

import time

from config.settings import settings
from generation.llm_client import chat_complete
from guardrails.checks import check_output
from pipeline.state import PipelineState

# ── Persona-specific tone tags (applied on top of affect base tag) ─────────────

_PERSONA_TONE_OVERRIDES: dict[str, dict[str, str]] = {
    "mia_chen": {
        "HAPPY":      "[TONE:WITTY_SARCASTIC]",
        "FRUSTRATED":  "[TONE:DIRECT_EMPATHETIC]",
    },
    "gerald_okafor": {
        "HAPPY":      "[TONE:WARM_FORMAL]",
        "FRUSTRATED":  "[TONE:MEASURED_EMPATHETIC]",
    },
    "arjun_mehta": {
        "HAPPY":      "[TONE:DIRECT_WARM]",
        "FRUSTRATED":  "[TONE:MINIMAL_DIRECT]",
    },
}


def run_primary(state: PipelineState) -> dict:
    return _run(state, tier="primary")


def run_fallback(state: PipelineState) -> dict:
    return _run(state, tier="fallback")


# def route_by_latency(state: PipelineState) -> str:
#     """Conditional edge after retrieval nodes."""
#     log = state.get("latency_log") or {}
#     elapsed = log.get("t_intent", 0.0) + log.get("t_retrieval", 0.0)
#     return "fallback" if elapsed > settings.fallback_latency_threshold else "primary"


# ── Core implementation ────────────────────────────────────────────────────────

def _run(state: PipelineState, tier: str) -> dict:
    t0 = time.perf_counter()

    profile = state["persona_profile"]
    user_id = state["user_id"]
    affect = (state.get("affect") or {}).get("emotion", "NEUTRAL")
    gen_cfg = state.get("generation_config") or {}
    chunks = state.get("retrieved_chunks") or []
    history = (state.get("session_history") or [])[-3:]   # last 3 turns only

    tone_tag = _resolve_tone_tag(user_id, affect, gen_cfg.get("tone_tag", "[TONE:DEFAULT]"))
    prompt = _build_prompt(profile, chunks, history, state["raw_query"], tone_tag, gen_cfg)

    candidates: list[str] = []
    for _ in range(settings.num_candidates):
        text = chat_complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=gen_cfg.get("max_tokens", settings.max_tokens_neutral) + 256,
            temperature=0.7,
            tier=tier,
        )
        candidates.append(text)

    selected = _rank_candidates(candidates, chunks, affect, profile)

    # Guardrail — replace with safe fallback if output breaks persona
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
        "candidates": candidates,
        "selected_response": selected,
        "llm_tier_used": tier,
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
) -> str:
    memory_block = "\n".join(f"  [{c['bucket']}] {c['text']}" for c in chunks) or "  (no memories retrieved)"
    history_block = "\n".join(f"  {h.get('role','?')}: {h.get('content','')}" for h in history) or "  (start of session)"
    style_exemplar = profile.get("style_exemplar", "")

    persona_mod = gen_cfg.get("persona_mod", "baseline")
    persona_instruction = {
        "amplify_quirks":    "Amplify your characteristic style and personality.",
        "suppress_humor":    "Be direct and supportive. Suppress humor.",
        "baseline":          "Use your natural communication style.",
        "add_confirmation":  "Add a clarifying question or confirmation at the end.",
    }.get(persona_mod, "Use your natural communication style.")

    return f"""\
You are {profile['name']}, an AAC device user with {profile['condition']}.
Communication style: {profile['style']}
{tone_tag}

Style exemplar — match this register:
  {style_exemplar}

Personal memories (use ONLY these for personal facts):
{memory_block}

Recent conversation:
{history_block}

Partner says: {query}

Instructions:
- Speak in first person as {profile['name']}.
- {persona_instruction}
- Keep response to 1-3 sentences.
- If the answer isn't in your memories, say "I don't know."
- Do NOT say "As an AI" or break persona.

Response:"""


def _rank_candidates(
    candidates: list[str],
    chunks: list[dict],
    affect: str,
    profile: dict,
) -> str:
    """
    Composite ranking: score = α·faithful + β·style + γ·affect_match
    Simple heuristic version — replace with NLI + cosine similarity for final eval.
    """
    if not candidates:
        return "I don't know."
    if len(candidates) == 1:
        return candidates[0]

    evidence_words = set(" ".join(c["text"] for c in chunks).lower().split())
    style_words = set(profile.get("style", "").lower().split())

    affect_positive_map = {
        "HAPPY":     ["great", "love", "enjoy", "happy", "fun"],
        "FRUSTRATED": ["okay", "fine", "sure", "yes", "no"],
        "NEUTRAL":   [],
        "SURPRISED": ["really", "oh", "interesting", "wow"],
    }
    affect_words = set(affect_positive_map.get(affect, []))

    def score(c: str) -> float:
        words = set(c.lower().split())
        faithful  = len(words & evidence_words) / max(len(words), 1)
        style_sim = len(words & style_words)    / max(len(words), 1)
        affect_m  = len(words & affect_words)   / max(len(words), 1)
        return (
            settings.rank_alpha * faithful
            + settings.rank_beta  * style_sim
            + settings.rank_gamma * affect_m
        )

    return max(candidates, key=score)
