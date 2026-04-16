# Multimodal AAC Chatbot — Project Guide

## What This Project Does

An AI chatbot that **speaks as an AAC user**, not to them. Given a user persona
(Mia, Gerald, or Arjun), it fuses real-time multimodal non-verbal signals with
personal memory retrieval to generate responses in that person's authentic voice.
Orchestrated as a **LangGraph stateful directed graph** across five layers.

---

## Architecture

```
frontend/                         React + Vite + TypeScript
  src/hooks/useSensing.ts         MediaPipe JS — affect, gesture, gaze, air-writing (browser-side)
  src/components/ChatPanel.tsx    Chat UI → POST /chat with sensing labels

backend/                          Python (conda env: aac-chatbot)
  main.py                         CLI entry point
  api/main.py                     FastAPI REST API
  pipeline/graph.py               LangGraph StateGraph (5 nodes + conditional edges)
    pipeline/nodes/intent.py        L2 — LLM + Pydantic intent routing
    pipeline/nodes/retrieval.py     L3 — FAISS + BGE retrieval (fast / full)
    pipeline/nodes/planner.py       L4 — expression-conditioned generation
    pipeline/nodes/feedback.py      L5 — MLflow logging + Bayesian priors
  sensing/                        L1 — MediaPipe face mesh, gesture, gaze, air writing (Python, CLI use)
  retrieval/                      FAISS ops, HDBSCAN clustering, Bayesian bucket priors
  generation/                     Multi-tier LLM client (vLLM primary / fallback / Ollama local)
  guardrails/                     Input + output safety checks
  config/                         Pydantic BaseSettings — all config in one place

data/                             Shared data (personas, FAISS indexes)
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
- **Browser-side sensing** — MediaPipe JS runs in React frontend, only classified
  labels (affect, gesture, gaze bucket) are sent to the backend API

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
# One-time setup
bash setup.sh

# CLI (local Ollama tier)
python -m backend.main --debug

# Full stack
uvicorn backend.api.main:app --reload    # FastAPI on :8000
pnpm --dir frontend dev                  # React on :7550
```

---

## Configuration

All config lives in [backend/config/settings.py](backend/config/settings.py) as Pydantic `BaseSettings`.
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

- **NEVER use local Ollama models** (e.g. `qwen3:8b`, `gemma3:1b`) — this machine
  is not powerful enough and will break. Always use cloud-backed models like
  `qwen3.5:397b-cloud` or `gpt-oss:20b-cloud` via Ollama, or vLLM tiers.
- **Adding a persona**: add to `PERSONAS` in `data/generate_users.py`, re-run it,
  then `python -m backend.retrieval.vector_store` to rebuild indexes
- **Changing LLM**: set `ACTIVE_LLM_TIER` in `.env` — no code changes needed
- **Extending sensing**: add module under `backend/sensing/`, wire output into
  `PipelineState` fields in `backend/pipeline/state.py`
- **Guardrail tuning**: edit signal lists in `backend/guardrails/checks.py`
- **Affect → generation mapping**: `_AFFECT_CONFIG` in `backend/pipeline/nodes/intent.py`
  and `_PERSONA_TONE_OVERRIDES` in `backend/pipeline/nodes/planner.py`
- FAISS indexes in `data/faiss_store/` are gitignored — rebuilt from source JSONs
- Frontend uses pnpm, Node 22+
