# Multi-tier LLM client — primary (vLLM) / fallback / local (Ollama), all OpenAI-compatible.
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from openai import OpenAI

from backend.config.settings import settings


@lru_cache(maxsize=3)
def _build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def get_client(tier: str | None = None) -> OpenAI:
    resolved = tier or settings.active_llm_tier

    if resolved == "primary":
        return _build_client(settings.primary_base_url, settings.primary_api_key)
    if resolved == "fallback":
        return _build_client(settings.fallback_base_url, settings.fallback_api_key)
    # local / default
    return _build_client(settings.local_base_url, settings.local_api_key)


def active_model(tier: str | None = None) -> str:
    resolved = tier or settings.active_llm_tier
    models = {
        "primary": settings.primary_model,
        "fallback": settings.fallback_model,
        "local": settings.local_model,
    }
    if resolved not in models:
        raise ValueError(
            f"Unknown LLM tier: '{resolved}'. Must be primary/fallback/local."
        )
    return models[resolved]


def _apply_no_think(messages: list[dict]) -> list[dict]:
    # Prepend /no_think to first user message (Ollama thinking suppression).
    result = list(messages)
    for i, msg in enumerate(result):
        if msg.get("role") == "user":
            result[i] = {**msg, "content": f"/no_think\n\n{msg['content']}"}
            break
    return result


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def chat_complete(
    messages: list[dict],
    max_tokens: int,
    tier: str | None = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> str:
    # Returns response text. Handles thinking mode and local-tier collapsing.
    resolved_tier = tier or settings.active_llm_tier

    # Local dev: no GCP server available — collapse all tiers to Ollama
    if settings.active_llm_tier == "local":
        resolved_tier = "local"
    model = active_model(resolved_tier)
    client = get_client(resolved_tier)

    patched_messages = messages
    extra_body: dict[str, Any] = kwargs.pop("extra_body", {})

    # Suppress thinking for models that think by default.
    if settings.thinking_mode == "suppress":
        if resolved_tier == "local":
            patched_messages = _apply_no_think(messages)
        else:
            extra_body = {
                **extra_body,
                "chat_template_kwargs": {"enable_thinking": False},
            }

    # Add thinking budget when enabled.
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
    raw = (resp.choices[0].message.content if resp.choices else "") or ""
    print(
        f"[llm_client] tier={resolved_tier} model={model} raw_len={len(raw)} raw={raw[:200]!r}"
    )

    if settings.thinking_mode in ("off", "strip"):
        raw = _strip_think_tags(raw)

    stripped = raw.strip()
    if not stripped:
        print(
            f"[llm_client] WARNING: empty response after strip. finish_reason={resp.choices[0].finish_reason if resp.choices else 'none'}"
        )
    return stripped


def warmup(tier: str | None = None) -> None:
    chat_complete(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
        tier=tier,
        temperature=0.0,
    )
