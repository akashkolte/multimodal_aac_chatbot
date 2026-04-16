# Multimodal AAC Chatbot

An AI chatbot that **speaks as an AAC user**, not to them. Given a persona (Mia, Gerald, or Arjun),
it fuses real-time multimodal non-verbal signals — facial expressions, hand gestures, gaze, and
air writing — with personal memory retrieval to generate responses in that person's authentic voice.

Built as a training-free, agentic RAG pipeline orchestrated via **LangGraph**.

---

## Table of Contents

- [What is AAC?](#what-is-aac)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [Personas](#personas)
- [Team](#team)

---

## What is AAC?

**Augmentative and Alternative Communication (AAC)** refers to tools and technologies that help
people who have difficulty with spoken or written communication — including individuals with
Cerebral Palsy, ALS, Autism Spectrum Disorder, and other conditions. This project gives AAC users
a personalized digital twin that communicates on their behalf.

---

## System Architecture

```
React Frontend (browser)                    Backend (Python)
  MediaPipe JS sensing ──┐
  Chat UI ───────────────┼── POST /chat ──► FastAPI ──► LangGraph Pipeline
  Webcam feed ───────────┘                                │
                                            L2 Intent ──► L3 Retrieval ──► L4 Generation ──► L5 Feedback
```

| Layer | Module | What it does |
|-------|--------|-------------|
| L1 | `frontend/src/hooks/useSensing.ts` | MediaPipe JS — affect, gesture, gaze, air writing (browser-side) |
| L2 | `backend/pipeline/nodes/intent.py` | LLM + Pydantic-validated intent routing |
| L3 | `backend/pipeline/nodes/retrieval.py` | FAISS + BGE embeddings + cross-encoder reranking |
| L4 | `backend/pipeline/nodes/planner.py` | Expression-conditioned response generation (Qwen3) |
| L5 | `backend/pipeline/nodes/feedback.py` | MLflow tracking + Bayesian bucket prior update |

The pipeline runs as a **LangGraph stateful directed graph** with conditional edges:
- FRUSTRATED affect → fast retrieval path (k=2, no reranker)
- Latency > 3.5s → fallback to smaller Qwen3-8B model

---

## Prerequisites

- Python **3.10+** (via conda)
- Node.js **22+** and **pnpm**
- [Ollama](https://ollama.com) installed locally for the `local` LLM tier
- A webcam (for live sensing; optional for CLI mode)

---

## Setup

```bash
git clone https://github.com/akashkolte/multimodal_aac_chatbot.git
cd multimodal_aac_chatbot
bash setup.sh
```

The setup script handles:
- Conda environment creation (`aac-chatbot`, Python 3.12)
- Python dependency installation
- `.env` file creation from template
- FAISS index building (downloads BGE models on first run)
- Ollama model pull
- Frontend dependency installation (pnpm)

---

## Configuration

All settings live in [backend/config/settings.py](backend/config/settings.py) and can be overridden via `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `ACTIVE_LLM_TIER` | `local` | `local` (Ollama) \| `primary` (vLLM GCP) \| `fallback` (Qwen3-8B) |
| `LOCAL_MODEL` | `qwen3:8b` | Ollama model name for local dev |
| `LOCAL_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `PRIMARY_BASE_URL` | *(GCP IP)* | vLLM server URL on GCP |
| `PRIMARY_MODEL` | `Qwen/Qwen3-30B-A3B` | Primary MoE model served via vLLM |
| `FALLBACK_LATENCY_THRESHOLD` | `3.5` | Seconds before falling back to smaller model |
| `MLFLOW_TRACKING_URI` | `mlruns` | Local MLflow storage path |

---

## Running the Project

### Full stack (recommended)

```bash
bash run.sh
```

This starts Ollama (if needed), FastAPI on `:8000`, and React on `:7550`.
Open [http://localhost:7550](http://localhost:7550) in your browser.

### CLI only

```bash
conda activate aac-chatbot
python -m backend.main --debug
```

### API only

```bash
conda activate aac-chatbot
uvicorn backend.api.main:app --reload
```

Example request:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "mia_chen", "query": "What do you like to do on weekends?"}'
```

---

## Project Structure

```
multimodal_aac_chatbot/
├── frontend/                      React + Vite + TypeScript
│   └── src/
│       ├── components/            Chat UI, webcam, sensing status
│       ├── hooks/                 useWebcam, useSensing (MediaPipe JS)
│       └── lib/                   API client, sensing classification, DTW
│
├── backend/                       Python (conda env: aac-chatbot)
│   ├── main.py                    CLI entry point
│   ├── api/main.py                FastAPI REST API
│   ├── config/settings.py         Pydantic BaseSettings
│   ├── pipeline/
│   │   ├── graph.py               LangGraph StateGraph
│   │   ├── state.py               PipelineState TypedDict
│   │   └── nodes/                 intent, retrieval, planner, feedback
│   ├── sensing/                   MediaPipe modules (Python, CLI use)
│   ├── retrieval/                 FAISS, BGE, HDBSCAN, bucket priors
│   ├── generation/llm_client.py   3-tier LLM client (vLLM / Ollama)
│   └── guardrails/checks.py      Input + output safety checks
│
├── data/
│   ├── users.json                 Persona index
│   ├── memories/                  Per-persona memory JSONs
│   └── faiss_store/               FAISS indexes (gitignored, rebuilt)
│
├── setup.sh                       One-time setup script
├── run.sh                         Start backend + frontend
├── requirements.txt               Python dependencies
└── .env.example                   Environment variable template
```

---

## Personas

| ID | Name | Condition | Style | Access |
|----|------|-----------|-------|--------|
| `mia_chen` | Mia Chen, 28 | Cerebral palsy | Witty, dry humour, short punchy sentences | Webcam head-tracking |
| `gerald_okafor` | Gerald Okafor, 61 | ALS (early-mid stage) | Formal, measured, eloquent | Eye-gaze device |
| `arjun_mehta` | Arjun Mehta, 17 | Autism (non-verbal) | Direct, routine-focused, Hindi-English code-switching | Tablet touch grid |

Each persona has 25 memory chunks across 5 buckets: `family`, `medical`, `hobbies`, `daily_routine`, `social`.

To add a new persona, edit `data/generate_users.py` and re-run `python -m backend.retrieval.vector_store`.

---

## TODO

- [ ] Add more dataset
- [ ] Reduce latency in intention
- [ ] Add more detailed todos

### Evals (`backend/evals/`)

Per-turn metrics returned in `ChatResponse.eval_scores` and rendered in the React debug panel.

| Metric | File | Status |
|--------|------|--------|
| Communication Efficiency | `efficiency.py` | Done — SLO check on `t_total` |
| Factual Faithfulness | `faithfulness.py` | Stub |
| Multimodal Alignment | `multimodal_alignment.py` | Stub |
| Perceived Authenticity | (frontend) | UI star rating; not persisted yet |

- [ ] **Faithfulness** — Load cross-encoder NLI model (e.g. `cross-encoder/nli-deberta-v3-small`),
  split response into sentences, check entailment against evidence chunks. Groundedness =
  fraction with max entailment > 0.5; hallucination rate = fraction with contradiction > 0.5
  and entailment < 0.3. Empty `chunks` → `no_evidence=True`.
- [ ] **Multimodal Alignment** — Rule-based (no model):
  - Affect → sentiment-word overlap (reuse `affect_positive_map` from planner)
  - Gesture → expected-word overlap (reuse `gesture_word_map` from planner)
  - Gaze → check whether retrieved chunks came from `gaze_bucket` and response references them
  - Overall = mean of non-None sub-scores
- [ ] **Authenticity** — Persist Likert ratings (currently client-side only). Add `POST /chat/rate`.

---

## Team

- **Akash Kolte** — akashjag@buffalo.edu
- **Shwetangi** — shwetang@buffalo.edu

University at Buffalo, SUNY

---

## License

All rights reserved. See the [LICENSE](LICENSE) file for details.
