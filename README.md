# Multimodal AAC Chatbot

An AI chatbot that **speaks as an AAC user**, not to them. Given a persona (Mia, Gerald, or Arjun),
it fuses real-time multimodal non-verbal signals — facial expressions, hand gestures, gaze, and
air writing — with personal memory retrieval to generate responses in that person's authentic voice.

Built as a training-free, agentic RAG pipeline — a plain-Python function chain
with two conditional branches (no LangGraph / LangChain), torch-tensor
retrieval (no FAISS), and JSONL turn logging (no MLflow).

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
  Chat UI ───────────────┼── POST /chat ──► FastAPI ──► run_pipeline()
  Webcam feed ───────────┘                                │
                                            L2 Intent ──► L3 Retrieval ──► L4 Generation ──► L5 Feedback
```

| Layer | Module | What it does |
|-------|--------|-------------|
| L1 | `frontend/src/hooks/useSensing.ts` | MediaPipe JS — affect, gesture, gaze, air writing (browser-side) |
| L2 | `backend/pipeline/nodes/intent.py` | Keyword-based intent routing (no LLM) |
| L3 | `backend/pipeline/nodes/retrieval.py` | BGE-small embeddings + torch tensor cosine search (mps/cuda/cpu) |
| L4 | `backend/pipeline/nodes/planner.py` | Expression-conditioned response generation (Qwen3) |
| L5 | `backend/pipeline/nodes/feedback.py` | JSONL turn logging + Bayesian bucket prior update |

The pipeline is a plain Python function chain with two conditional branches:
- FRUSTRATED affect → fast retrieval path (k=2)
- Latency > 3.5s → fallback to smaller Qwen3-8B model

---

## Prerequisites

- Python **3.10+** (via conda)
- Node.js **22+** and **pnpm**
- An [Ollama Cloud](https://ollama.com) account — both LLM tiers hit
  cloud-hosted models; no local Ollama daemon required
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
- Vector index building (downloads BGE-small embedder on first run, saves
  per-user `vectors.pt` under `data/faiss_store/`)
- Frontend dependency installation (pnpm)

---

## Configuration

All settings live in [backend/config/settings.py](backend/config/settings.py) and can be overridden via `.env`.

| Variable | Default | Description |
|----------|---------|-------------|
| `ACTIVE_LLM_TIER` | `primary` | `primary` \| `fallback` |
| `PRIMARY_MODEL` | `gemma4:31b-cloud` | Ollama Cloud model for primary tier |
| `FALLBACK_MODEL` | `gemma4:31b-cloud` | Ollama Cloud model for fallback tier (smaller/faster) |
| `PRIMARY_BASE_URL` | `http://localhost:11434/v1` | Ollama-compatible endpoint |
| `FALLBACK_LATENCY_THRESHOLD` | `3.5` | Seconds before falling back to smaller model |
| `LOGS_DIR` | `logs` | Where per-turn JSONL logs are written |

---

## Running the Project

### Full stack (recommended)

```bash
bash run.sh
```

This starts FastAPI on `:8000` and React on `:7550`.
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
│   │   ├── graph.py               run_pipeline() — plain function chain
│   │   ├── state.py               PipelineState TypedDict
│   │   └── nodes/                 intent, retrieval, planner, feedback
│   ├── sensing/labels.py          GESTURE_TO_TAG (sensing runs in browser)
│   ├── retrieval/                 BGE embeddings (torch tensor) + bucket priors
│   ├── generation/llm_client.py   2-tier Ollama Cloud LLM client (primary/fallback)
│   └── guardrails/checks.py      Input + output safety checks
│
├── data/
│   ├── users.json                 Persona index
│   ├── memories/                  Per-persona memory JSONs
│   └── faiss_store/               vectors.pt + meta.json (gitignored, rebuilt)
├── logs/                          Per-turn JSONL logs (gitignored)
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

From the spec (pages 10–11). Tags: **[Core]** = must do, **[Bonus]** = nice to have, **[Eval]** = for the grade.

Heads up: all camera/sensing stuff is in the frontend (MediaPipe JS). Backend just gets the labels (`affect`, `gesture_tag`, `gaze_bucket`). The `backend/sensing/` python modules are dead code.

### Dataset

- [ ] **[Core]** Memories are only autobiographical narratives right now. Need more variety:
  - [ ] social media posts (voice-matched, synth with LLM)
  - [ ] past chat logs (synth with LLM)
  - [ ] update the generator script + rebuild faiss
  - [ ] tag chunks by type so retriever knows what it pulled
- [ ] **[Core]** Write down the data schema somewhere so evals can reuse it

### Sensing (frontend)

- [ ] **[Core]** Head-nod / sharp tilt = "I don't like that". Different from frustrated affect.
  - [ ] send a `dissatisfaction_signal` flag with the chat request
  - [ ] when set, planner returns a "did you mean X or Y?" instead of an answer (the spec's "Turnaround Option")
- [ ] **[Core]** Smile / positive affect should actually change the wording (more positive lexicon), not just be metadata. Right now it's annotated in the prompt but we never checked if the LLM is doing anything with it — probably need a stronger constraint or example in the prompt
- [ ] **[Core]** Air-writing is treated as raw text appended to the query. Spec wants it as a stylistic constraint too — should it bias tone, or stay query-only? Decide and document
- [ ] **[Bonus]** Voice + air-writing conflict resolution. Capture short voice (Web Speech API), compare to air-written intent, send a `resolved_intent`
- [ ] thumbs-up only changes the prompt today — should also boost affirmative candidates in the reranker

### Intent decomposition

> Current state: routing is keyword-based, not LLM-based. The original LLM router (Pydantic-validated JSON) kept emitting the wrong shape with `gemma4:31b-cloud` and hitting the `max_tokens` truncation — 3 retries + hard fallback on every turn, ~30s of dead latency before generation. The keyword router (5 buckets matched against word lists in `intent.py`) handles the demo personas and adds ~0ms. Trade-off: stuck with the 5 hardcoded buckets (`family`, `medical`, `hobbies`, `daily_routine`, `social`) and can't tell `OPEN_DOMAIN` from `PERSONAL`. Fine for now since all personas only have personal memories. Revisit when Ollama Cloud ships `response_format=json_schema` or we add a tiny local classifier.

- [ ] **[Core]** Personal / Contextual / Open-domain all hit the same FAISS index right now. Make them actually go different places — open-domain → web search (or stub), contextual → session memory
- [ ] intent node is slow. Cache the prompt, use a tiny model for routing, parallelise the sub-queries

### Retrieval

- [ ] **[Bonus]** Bucket priors only live for the session. Persist them per user
- [ ] **[Bonus]** Latency fallback only switches LLM tier. Add more steps:
  - drop reranker if retrieval is slow
  - return a canned response if we blow the budget entirely
  - threshold is 3.5s, spec says 6s — pick one
- [ ] **[Scale]** past ~100k chunks per user, swap torch matmul for `hnswlib`; add a reranker if top_k grows past ~30

### Generation

- [ ] **[Core]** API returns one response. Should return multiple candidates so the user can pick (and so the next item works)
- [ ] **[Core]** Frontend needs a candidate picker — show all the options, let the user click one, send the selection back
- [ ] **[Bonus]** When user picks a candidate, save the `(query, picked)` pair to a side faiss index and check it first next turn

### Evals

Live per-turn scores show up in the `EvalPanel`. State:

| Metric | Status |
|--------|--------|
| Efficiency | works (SLO check on `t_total`) |
| Faithfulness | stub, returns 0 |
| Multimodal alignment | stub, returns 0 |
| Authenticity | star rating in UI but not saved |

- [ ] **[Eval]** Faithfulness — actually check if the response is grounded in what we retrieved. NLI model, sentence-level. If we didn't retrieve anything, flag `no_evidence` instead of pretending we scored it
- [ ] **[Eval]** Efficiency — per-turn SLO check is done, but for the writeup we need aggregate latency: p50/p95 across a fixed query set, broken out by LLM tier. Spec target is < 6s
- [ ] **[Eval]** Multimodal alignment — does the response actually reflect the gesture/affect/gaze? Don't need a model for this, just reuse the word maps the planner already has. Gaze one is trickier — check whether the chunks we ended up using came from the bucket the user was looking at
- [ ] **[Eval]** Authenticity — the Likert stars are wired up in the UI but go nowhere. Save them, log them with the turn so we can actually look at them later
- [ ] **[Eval]** For the live in-class eval: figure out the actual session — who rates (partners + experts per spec), how many turns each, what gets shown to them. The Likert form is the easy part; the protocol isn't written down anywhere
- [ ] **[Eval]** Need an offline version of all three model-driven evals (faithfulness / alignment / efficiency). Aggregate numbers across a fixed query set per persona for the writeup

### Cleanup

- [ ] move the affect→tone / persona override dicts out of code into a yaml
- [x] delete `backend/sensing/` (dead code, sensing is in frontend) — done, only `labels.py` remains

---

## Team

- **Akash Kolte** — akashjag@buffalo.edu
- **Shwetangi** — shwetang@buffalo.edu

University at Buffalo, SUNY

---

## License

All rights reserved. See the [LICENSE](LICENSE) file for details.
