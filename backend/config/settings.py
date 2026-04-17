from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    vector_store_dir: Path = Path("data/vector_store")
    memories_dir: Path = Path("data/memories")
    users_json: Path = Path("data/users.json")
    logs_dir: Path = Path("logs")

    # ── Retrieval ────────────────────────────────────────────────────────────
    embed_model: str = "BAAI/bge-small-en-v1.5"
    retrieval_top_k: int = 5
    retrieval_rerank_k: int = 3
    retrieval_fast_k: int = 2  # used when affect == FRUSTRATED
    # Minimum cosine score for a chunk to be used in turnaround re-retrieval.
    # Below this, we'd rather fall back to original chunks than serve clearly
    # off-topic memories just to "look different."
    turnaround_min_score: float = 0.45

    rerank_enabled: bool = True
    rerank_pool_k: int = 12  # wider pre-rerank fetch per personal sub-intent
    rerank_fast_pool_k: int = 8  # smaller pool on the FRUSTRATED fast path
    rerank_lambda: float = 0.7  # MMR: relevance vs diversity (1.0 = pure cosine)
    rerank_history_turns: int = 2  # last-N user turns folded into context vector
    rerank_query_weight: float = 0.7  # current query weight vs history mean

    # LLM tiers — both hit Ollama Cloud via OpenAI-compatible endpoint.
    # Same model on both tiers for now; swap one when a larger cloud model
    # is provisioned and the latency-fallback should branch.
    primary_model: str = "gemma4:31b-cloud"
    primary_base_url: str = "http://localhost:11434/v1"
    primary_api_key: str = "ollama"

    fallback_model: str = "gemma4:31b-cloud"
    fallback_base_url: str = "http://localhost:11434/v1"
    fallback_api_key: str = "ollama"

    # Active tier: "primary" | "fallback"
    active_llm_tier: str = "primary"

    # off | strip | full | suppress
    thinking_mode: str = "off"
    thinking_token_budget: int = 4096
    fallback_latency_threshold: float = 3.5  # seconds before tier fallback

    # ── Generation ────────────────────────────────────────────────────────────
    max_tokens_happy: int = 150
    max_tokens_neutral: int = 100
    max_tokens_frustrated: int = 60
    max_tokens_surprised: int = 80

    # ── Sensing ───────────────────────────────────────────────────────────────
    affect_ema_alpha: float = 0.3  # exponential moving average smoothing
    gaze_dwell_threshold_s: float = 1.5
    air_write_velocity_start: int = 15  # px/frame — stroke begin threshold
    air_write_velocity_end: int = 5  # px/frame — stroke end threshold
    air_write_end_gap_ms: int = 200  # ms of stillness to end a stroke
    conflict_overlap_ms: int = 500  # audio + gesture co-occurrence window

    # ── Evaluation ────────────────────────────────────────────────────────────
    slo_target_s: float = 6.0  # max acceptable response latency (seconds)
    evals_enabled: bool = True
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    faithfulness_threshold: float = (
        0.5  # entailment prob for a sentence to count as grounded
    )


settings = Settings()
