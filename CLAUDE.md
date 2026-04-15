# Multimodal AAC Chatbot — Project Guide

## What This Project Does

An AI chatbot that **speaks as an AAC user**, not to them. Given a user persona
(Mia, Gerald, or Arjun), it fuses real-time multimodal non-verbal signals with
personal memory retrieval to generate responses in that person's authentic voice.
Orchestrated as a **LangGraph stateful directed graph** across five layers.

---

## Architecture

```
main.py  /  api/main.py  /  ui/app.py
  └── pipeline/graph.py              ← LangGraph StateGraph (5 nodes + cond. edges)
        ├── pipeline/nodes/intent.py      L2 — LLM + Pydantic intent routing
        ├── pipeline/nodes/retrieval.py   L3 — FAISS + BGE retrieval (fast / full)
        ├── pipeline/nodes/planner.py     L4 — expression-conditioned generation
        └── pipeline/nodes/feedback.py    L5 — MLflow logging + Bayesian priors

sensing/          L1 — MediaPipe face mesh, gesture, gaze, air writing
retrieval/        FAISS ops, HDBSCAN clustering, Bayesian bucket priors
generation/       Multi-tier LLM client (vLLM primary / fallback / Ollama local)
guardrails/       Input + output safety checks
config/           Pydantic BaseSettings — all config in one place
```

## Key Design Decisions

- **LangGraph** orchestrates the pipeline as a stateful directed graph with
  conditional edges (affect → fast/full retrieval; latency → primary/fallback LLM)
- **BGE-small-en-v1.5** for embeddings (beats MiniLM on MTEB at same speed)
- **BGE-reranker-v2-m3** cross-encoder — multilingual, handles Arjun's Hindi
- **FAISS IndexFlatIP** with L2-normalised vectors (inner product = cosine sim)
- **Qwen3-30B-A3B** MoE via vLLM — 3B active params/token, sub-3s on T4
- **Three-tier LLM fallback**: primary (vLLM GCP) → fallback (Qwen3-8B) → local (Ollama)
- **Pydantic-validated** LLM routing output — LangGraph retries on schema failures
- **Expression-conditioned response shaping** — affect steers tone, retrieval depth,
  and candidate ranking (not just metadata annotation)
- **Bayesian bucket priors** — session-level P(bucket) updated after each accepted turn

---

## Personas

| ID | Name | Condition | Access |
|----|------|-----------|--------|
| `mia_chen` | Mia Chen, 28 | Cerebral palsy | Webcam head-tracking |
| `gerald_okafor` | Gerald Okafor, 61 | ALS (early-mid) | Eye-gaze device |
| `arjun_mehta` | Arjun Mehta, 17 | Autism (non-verbal) | Tablet touch grid |

25 memory chunks each (5 buckets × 5 memories). Arjun code-switches Hindi/English.

---

## How to Run

```bash
# One-time setup: rebuild FAISS indexes with BGE embedder
python -m retrieval.vector_store

# CLI (local Ollama tier, set ACTIVE_LLM_TIER=local in .env)
python main.py --debug

# Full stack
uvicorn api.main:app --reload        # FastAPI on :8000
streamlit run ui/app.py              # Streamlit on :8501
```

---

## Configuration

All config lives in [config/settings.py](config/settings.py) as Pydantic `BaseSettings`.
Copy `.env.example` → `.env` and set:

- `ACTIVE_LLM_TIER` — `local` (dev) | `primary` (GCP A100) | `fallback` (Qwen3-8B)
- `PRIMARY_BASE_URL` — vLLM server address on GCP
- `MLFLOW_TRACKING_URI` — where MLflow stores runs (default: `mlruns/`)

---

## Data Files

| Path | Purpose |
|------|---------|
| `data/users.json` | Flat user index (id, name, condition, style) |
| `data/memories/<uid>.json` | Full persona JSON with bucketed memories |
| `data/faiss_store/<uid>/` | FAISS index + metadata — **rebuild after any persona edit** |
| `data/generate_users.py` | Regenerates memories + users.json |

---

## Development Notes

- **Adding a persona**: add to `PERSONAS` in `data/generate_users.py`, re-run it,
  then `python -m retrieval.vector_store` to rebuild indexes
- **Changing LLM**: set `ACTIVE_LLM_TIER` in `.env` — no code changes needed
- **Extending sensing**: add module under `sensing/`, wire output into
  `PipelineState` fields in `pipeline/state.py`
- **Guardrail tuning**: edit signal lists in `guardrails/checks.py`
- **Affect → generation mapping**: `_AFFECT_CONFIG` in `pipeline/nodes/intent.py`
  and `_PERSONA_TONE_OVERRIDES` in `pipeline/nodes/planner.py`
- The `.venv/` directory is local — do not read or modify files inside it
- FAISS indexes in `data/faiss_store/` are gitignored — rebuilt from source JSONs
