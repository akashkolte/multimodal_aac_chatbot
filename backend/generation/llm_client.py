# Two-tier LLM client — primary / fallback over OpenAI-compatible HTTP.
# Supports Ollama (local) and NVIDIA NIM (https://integrate.api.nvidia.com/v1).
import re
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from backend.config.settings import settings

_console = Console(stderr=False)


@lru_cache(maxsize=2)
def _build_client(base_url: str, api_key: str) -> OpenAI:
    clean_key = api_key[7:] if api_key.startswith("Bearer ") else api_key
    return OpenAI(base_url=base_url, api_key=clean_key)


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


def _is_ollama(base_url: str) -> bool:
    """Return True when the endpoint is a local Ollama instance."""
    return "localhost:11434" in base_url or "127.0.0.1:11434" in base_url


def _apply_no_think(messages: list[dict]) -> list[dict]:
    # Prepend /no_think to first user message — Ollama-only thinking suppression.
    # Do NOT apply this to NVIDIA NIM or other providers.
    result = list(messages)
    for i, msg in enumerate(result):
        if msg.get("role") == "user":
            result[i] = {**msg, "content": f"/no_think\n\n{msg['content']}"}
            break
    return result


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _print_response(
    tier: str,
    model: str,
    raw: str,
    finish_reason: str | None,
    tokens_used: int | None,
) -> None:
    """Print a rich-formatted summary of the LLM response to the console."""
    meta = Text()
    meta.append("Tier: ",      style="bold dim")
    meta.append(f"{tier}   ",  style="cyan")
    meta.append("Model: ",     style="bold dim")
    meta.append(f"{model}   ", style="cyan")
    meta.append("Finish: ",    style="bold dim")
    meta.append(f"{finish_reason or 'n/a'}   ", style="green" if finish_reason == "stop" else "yellow")
    meta.append("Tokens: ",    style="bold dim")
    meta.append(str(tokens_used or "n/a"), style="cyan")

    body = Text(raw or "(empty)", style="white" if raw else "red")

    _console.print(Panel(
        body,
        title=meta,
        title_align="left",
        border_style="bright_blue",
        padding=(0, 1),
    ))


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

    if settings.thinking_mode == "suppress" and _is_ollama(settings.primary_base_url):
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
    finish_reason = resp.choices[0].finish_reason if resp.choices else None
    tokens_used = getattr(resp.usage, "completion_tokens", None) if resp.usage else None
    _print_response(resolved_tier, model, raw, finish_reason, tokens_used)

    if settings.thinking_mode in ("off", "strip"):
        raw = _strip_think_tags(raw)

    stripped = raw.strip()
    if not stripped:
        _console.print(Panel(
            f"[bold red]WARNING:[/bold red] Empty response after strip.  "
            f"finish_reason=[yellow]{finish_reason}[/yellow]",
            border_style="red",
            title="[red]llm_client[/red]",
        ))
    return stripped


def chat_complete_stream(
    messages: list[dict],
    max_tokens: int,
    tier: str | None = None,
    temperature: float = 0.7,
    **kwargs: Any,
) -> Iterator[str]:
    """Yield token deltas as they arrive. Thinking-mode stripping is applied
    post-hoc on the buffered text by the caller — streaming <think>…</think>
    into the UI would confuse the picker anyway.
    """
    resolved_tier = tier or settings.active_llm_tier
    model = active_model(resolved_tier)
    client = get_client(resolved_tier)

    patched_messages = messages
    extra_body: dict[str, Any] = kwargs.pop("extra_body", {})

    if settings.thinking_mode == "suppress" and _is_ollama(settings.primary_base_url):
        patched_messages = _apply_no_think(messages)

    effective_max_tokens = max_tokens
    if settings.thinking_mode in ("strip", "full"):
        effective_max_tokens = max_tokens + settings.thinking_token_budget

    stream = client.chat.completions.create(
        model=model,
        messages=patched_messages,
        max_tokens=effective_max_tokens,
        temperature=temperature,
        stream=True,
        extra_body=extra_body or None,
        **kwargs,
    )
    buf: list[str] = []
    finish_reason: str | None = None
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        piece = getattr(delta, "content", None) or ""
        if piece:
            buf.append(piece)
            yield piece
        # capture finish reason from the last chunk
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    _print_response(resolved_tier, model, "".join(buf), finish_reason, len(buf))


def finalize_streamed(text: str) -> str:
    """Apply the same post-processing chat_complete does once a stream is done."""
    if settings.thinking_mode in ("off", "strip"):
        text = _strip_think_tags(text)
    return text.strip()


def warmup(tier: str | None = None) -> None:
    chat_complete(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=5,
        tier=tier,
        temperature=0.0,
    )
