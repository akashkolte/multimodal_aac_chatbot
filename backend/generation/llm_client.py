# Two-tier LLM client — primary / fallback, both Ollama Cloud over OpenAI-compatible HTTP.
import re
from functools import lru_cache
from typing import Any

from openai import OpenAI

from backend.config.settings import settings


@lru_cache(maxsize=2)
def _build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def get_client(tier: str | None = None) -> OpenAI:
    resolved = tier or settings.active_llm_tier
    if resolved == "fallback":
        return _build_client(settings.fallback_base_url, settings.fallback_api_key)
    return _build_client(settings.primary_base_url, settings.primary_api_key)


def active_model(tier: str | None = None) -> str:
    resolved = tier or settings.active_llm_tier
    models = {"primary": settings.primary_model, "fallback": settings.fallback_model}
    if resolved not in models:
        raise ValueError(f"Unknown LLM tier: '{resolved}'. Must be primary/fallback.")
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
    resolved_tier = tier or settings.active_llm_tier
    model = active_model(resolved_tier)
    client = get_client(resolved_tier)

    patched_messages = messages
    extra_body: dict[str, Any] = kwargs.pop("extra_body", {})

    if settings.thinking_mode == "suppress":
        patched_messages = _apply_no_think(messages)

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
