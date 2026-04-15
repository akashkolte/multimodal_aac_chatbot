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
Webcam (L1: sensing) → Intent Decomposition (L2) → Retrieval (L3) → Generation (L4) → Feedback (L5)
```

| Layer | Module | What it does |
|-------|--------|-------------|
| L1 | `sensing/` | MediaPipe face mesh, hand gestures, gaze tracking, air writing |
| L2 | `pipeline/nodes/intent.py` | LLM + Pydantic-validated intent routing |
| L3 | `pipeline/nodes/retrieval.py` | FAISS + BGE embeddings + cross-encoder reranking |
| L4 | `pipeline/nodes/planner.py` | Expression-conditioned response generation (Qwen3) |
| L5 | `pipeline/nodes/feedback.py` | MLflow tracking + Bayesian bucket prior update |

The pipeline runs as a **LangGraph stateful directed graph** with conditional edges:
- FRUSTRATED affect → fast retrieval path (k=2, no reranker)
- Latency > 3.5s → fallback to smaller Qwen3-8B model

---

## Prerequisites

- Python **3.10 – 3.12** (Python 3.14 has a known Pydantic v1 incompatibility warning — functional but noisy)
- [Ollama](https://ollama.com) installed locally for the `local` LLM tier
- A webcam (required for the live sensing layer; optional for CLI mode)
- Git

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/akashkolte/multimodal_aac_chatbot.git
cd multimodal_aac_chatbot
```

### 2. Check out the active branch

```bash
git checkout akash/v1
```

### 3. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> This installs LangGraph, FAISS, sentence-transformers (BGE), FastAPI, Streamlit, MLflow,
> MediaPipe, and all other dependencies.

### 5. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
ACTIVE_LLM_TIER=local          # use Ollama on your machine for dev
```

See [Configuration](#configuration) for all options.

### 6. Pull the local LLM model (Ollama)

```bash
ollama pull qwen3:8b
```

> Make sure Ollama is running (`ollama serve`) before starting the chatbot.

### 7. Build FAISS indexes

The persona memory indexes must be built once with the BGE embedder before first run:

```bash
python -m retrieval.vector_store
```

Expected output:
```
Building index for arjun_mehta … Saved 25 chunks
Building index for gerald_okafor … Saved 25 chunks
Building index for mia_chen … Saved 25 chunks
All indexes built.
```

> You must re-run this step whenever you add or edit persona memory files.

---

## Configuration

All settings live in [config/settings.py](config/settings.py) and can be overridden via `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `ACTIVE_LLM_TIER` | `local` | `local` (Ollama) \| `primary` (vLLM GCP) \| `fallback` (Qwen3-8B) |
| `LOCAL_MODEL` | `qwen3:8b` | Ollama model name for local dev |
| `LOCAL_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `PRIMARY_BASE_URL` | *(GCP IP)* | vLLM server URL on GCP (set when using cloud tier) |
| `PRIMARY_MODEL` | `Qwen/Qwen3-30B-A3B` | Primary MoE model served via vLLM |
| `FALLBACK_LATENCY_THRESHOLD` | `3.5` | Seconds before falling back to smaller model |
| `MLFLOW_TRACKING_URI` | `mlruns` | Local MLflow storage path |
| `MLFLOW_EXPERIMENT` | `aac-chatbot` | MLflow experiment name |

---

## Running the Project

### Option A — CLI (simplest, no webcam needed)

```bash
python main.py
```

With debug latency output:
```bash
python main.py --debug
```

Select a specific persona and LLM tier:
```bash
python main.py --user mia_chen --tier local
```

### Option B — Full stack (FastAPI + Streamlit UI)

Start the API server in one terminal:
```bash
uvicorn api.main:app --reload --port 8000
```

Start the Streamlit frontend in another terminal:
```bash
streamlit run ui/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

The UI includes:
- Persona selector
- Affect override controls (simulate webcam for testing)
- Live chat interface
- Per-turn latency breakdown panel

### Option C — API only (for integration / testing)

```bash
uvicorn api.main:app --reload
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
│
├── config/
│   └── settings.py            # All config via Pydantic BaseSettings
│
├── data/
│   ├── generate_users.py      # Regenerates persona memories + users.json
│   ├── users.json             # Flat user index
│   ├── memories/              # Per-persona memory JSON files
│   └── faiss_store/           # Built FAISS indexes (gitignored, rebuild locally)
│
├── sensing/                   # L1 — multimodal input
│   ├── face_mesh.py           # MediaPipe affect detection (MAR/EAR/BRI/LCP)
│   ├── gesture.py             # Hand gesture classifier
│   ├── gaze.py                # Gaze-based bucket activation (bonus)
│   └── air_writing.py         # DTW air-writing stroke classifier (bonus)
│
├── pipeline/                  # LangGraph orchestration
│   ├── state.py               # Typed PipelineState (TypedDict)
│   ├── graph.py               # Graph definition + conditional edges
│   └── nodes/
│       ├── intent.py          # L2 — LLM + Pydantic routing
│       ├── retrieval.py       # L3 — fast + full retrieval paths
│       ├── planner.py         # L4 — expression-conditioned generation
│       └── feedback.py        # L5 — MLflow + Bayesian prior update
│
├── retrieval/
│   ├── vector_store.py        # FAISS ops with BGE-small-en-v1.5
│   ├── clustering.py          # HDBSCAN semantic bucketing
│   └── bucket_priors.py       # Bayesian session priors
│
├── generation/
│   └── llm_client.py          # 3-tier LLM client (vLLM / Ollama)
│
├── guardrails/
│   └── checks.py              # Input + output safety checks
│
├── api/
│   └── main.py                # FastAPI backend
│
├── ui/
│   └── app.py                 # Streamlit frontend
│
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
└── CLAUDE.md                  # Developer notes (AI assistant context)
```

---

## Personas

| ID | Name | Condition | Style | Access |
|----|------|-----------|-------|--------|
| `mia_chen` | Mia Chen, 28 | Cerebral palsy | Witty, dry humour, short punchy sentences | Webcam head-tracking |
| `gerald_okafor` | Gerald Okafor, 61 | ALS (early-mid stage) | Formal, measured, eloquent | Eye-gaze device |
| `arjun_mehta` | Arjun Mehta, 17 | Autism (non-verbal) | Direct, routine-focused, Hindi-English code-switching | Tablet touch grid |

Each persona has 25 memory chunks across 5 buckets: `family`, `medical`, `hobbies`, `daily_routine`, `social`.

To add a new persona, edit `data/generate_users.py` and re-run `python -m retrieval.vector_store`.

---

## Team

- **Akash Kolte** — akashjag@buffalo.edu
- **Shwetangi** — shwetang@buffalo.edu

University at Buffalo, SUNY

---

## License

All rights reserved. See the [LICENSE](LICENSE) file for details.
