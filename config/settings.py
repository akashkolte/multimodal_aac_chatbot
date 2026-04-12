from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Paths ──────────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    faiss_store_dir: Path = Path("data/faiss_store")
    memories_dir: Path = Path("data/memories")
    users_json: Path = Path("data/users.json")

    # ── Retrieval models ───────────────────────────────────────────────────────
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    retrieval_top_k: int = 5
    retrieval_rerank_k: int = 3
    retrieval_fast_k: int = 2          # used when affect == FRUSTRATED

    # ── LLM tiers ─────────────────────────────────────────────────────────────
    # Tier 1 — primary (Qwen3-30B-A3B via vLLM on GCP)
    primary_model: str = "Qwen/Qwen3-30B-A3B"
    primary_base_url: str = "http://localhost:8000/v1"
    primary_api_key: str = "token-abc"          # vLLM default

    # Tier 2 — fallback dense model (Qwen3-8B via vLLM, same server)
    fallback_model: str = "Qwen/Qwen3-8B"
    fallback_base_url: str = "http://localhost:8000/v1"

    # Tier 3 — local dev (Ollama on MacBook M2)
    local_model: str = "qwen3:8b"
    local_base_url: str = "http://localhost:11434/v1"
    local_api_key: str = "ollama"

    # Active tier: "primary" | "fallback" | "local"
    active_llm_tier: str = "local"

    # Wall-clock threshold (seconds) that triggers fallback within a turn
    fallback_latency_threshold: float = 3.5

    # ── Generation ────────────────────────────────────────────────────────────
    max_tokens_happy: int = 150
    max_tokens_neutral: int = 100
    max_tokens_frustrated: int = 60
    max_tokens_surprised: int = 80
    num_candidates: int = 2            # responses generated per turn for ranking

    # ── Sensing ───────────────────────────────────────────────────────────────
    affect_ema_alpha: float = 0.3      # exponential moving average smoothing
    gaze_dwell_threshold_s: float = 1.5
    air_write_velocity_start: int = 15  # px/frame — stroke begin threshold
    air_write_velocity_end: int = 5     # px/frame — stroke end threshold
    air_write_end_gap_ms: int = 200     # ms of stillness to end a stroke
    conflict_overlap_ms: int = 500      # audio + gesture co-occurrence window

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment: str = "aac-chatbot"

    # ── Candidate ranking weights (Eq. 2 in proposal) ─────────────────────────
    rank_alpha: float = 0.4            # faithfulness weight
    rank_beta: float = 0.3             # style similarity weight
    rank_gamma: float = 0.3            # affect-match weight


settings = Settings()
