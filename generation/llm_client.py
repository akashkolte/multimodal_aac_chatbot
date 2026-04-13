"""
Multi-tier LLM client (proposal §5.6).

All three tiers expose the same OpenAI-compatible API, so only the
base_url + model name change — no code-path differences downstream.

Tier 1 — primary:  Qwen3-30B-A3B via vLLM on GCP (A100 / T4)
Tier 2 — fallback: Qwen3-8B via vLLM on same server (latency > 3.5 s)
Tier 3 — local:    Qwen3-8B via Ollama on MacBook M2 (dev / offline)

Active tier is controlled by settings.active_llm_tier or the `tier`
argument passed explicitly by the planner node.

Thinking mode is controlled by settings.thinking_mode:
  "off"   — prepend /no_think (Ollama) or chat_template_kwargs (vLLM)
  "strip" — let the model think, but strip <think>…</think> from output
  "full"  — return everything including <think> blocks
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from openai import OpenAI

from config.settings import settings


@lru_cache(maxsize=3)
def _build_client(base_url: str, api_key: str) -> OpenAI:
    """One cached OpenAI client per (base_url, api_key) pair."""
    return OpenAI(base_url=base_url, api_key=api_key)


def get_client(tier: str | None = None) -> OpenAI:
    """
    Return the OpenAI-compatible client for the requested tier.

    Args:
        tier: "primary" | "fallback" | "local" | None (uses settings.active_llm_tier)
    """
    resolved = tier or settings.active_llm_tier

    if resolved == "primary":
        return _build_client(settings.primary_base_url, settings.primary_api_key)
    if resolved == "fallback":
        return _build_client(settings.fallback_base_url, settings.primary_api_key)
    # local / default
    return _build_client(settings.local_base_url, settings.local_api_key)


def active_model(tier: str | None = None) -> str:
    """Return the model name string for the given tier."""
    resolved = tier or settings.active_llm_tier
    return {
        "primary":  settings.primary_model,
        "fallback": settings.fallback_model,
        "local":    settings.local_model,
    }[resolved]


def _apply_no_think(messages: list[dict]) -> list[dict]:
    """
    Prepend /no_think to the first user message.
    This is the Ollama-compatible way to suppress thinking mode.
    """
    result = list(messages)
    for i, msg in enumerate(result):
        if msg.get("role") == "user":
            result[i] = {**msg, "content": f"/no_think\n\n{msg['content']}"}
            break
    return result


def _strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def chat_complete(
    messages: list[dict],
    max_tokens: int,
    tier: str | None = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> str:
    """
    Model-agnostic chat completion. Returns the response text directly.

    Thinking mode behaviour is controlled entirely by settings.thinking_mode:
      "off"   — suppress thinking via /no_think (Ollama) or extra_body (vLLM)
      "strip" — allow thinking but remove <think> tags from the response
      "full"  — return the raw response including any <think> blocks

    In local dev mode (active_llm_tier="local"), all tier requests are
    redirected to Ollama — there is no separate fallback server locally.
    """
    resolved_tier = tier or settings.active_llm_tier

    # Local dev: no GCP server available — collapse all tiers to Ollama
    if settings.active_llm_tier == "local":
        resolved_tier = "local"
    model = active_model(resolved_tier)
    client = get_client(resolved_tier)

    patched_messages = messages
    extra_body: dict[str, Any] = kwargs.pop("extra_body", {})

    # "suppress" = actively inject /no_think or vLLM flag for models
    # like Qwen3 that think by default and need explicit suppression.
    if settings.thinking_mode == "suppress":
        if resolved_tier == "local":
            patched_messages = _apply_no_think(messages)
        else:
            extra_body = {**extra_body, "chat_template_kwargs": {"enable_thinking": False}}

    # When thinking is enabled (strip/full), add budget so the model
    # has room to reason without truncating the actual answer.
    effective_max_tokens = max_tokens
    if settings.thinking_mode in ("strip", "full"):
        effective_max_tokens = max_tokens + settings.thinking_token_budget

    resp = client.chat.completions.create(
        model=model,
        messages=patched_messages,
        max_tokens=effective_max_tokens,
        temperature=temperature,
        extra_body=extra_body or None,
        **kwargs,
    )
    raw = resp.choices[0].message.content or ""

    if settings.thinking_mode in ("off", "strip"):
        raw = _strip_think_tags(raw)

    return raw.strip()


def warmup(tier: str | None = None) -> None:
    """Send a minimal prompt to pre-load the model and warm KV cache."""
    chat_complete(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
        tier=tier,
        temperature=0.0,
    )
