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

<<<<<<< Updated upstream
From the spec (pages 10–11). Tags: **[Core]** = must do, **[Bonus]** = nice to have, **[Eval]** = for the grade.
=======
Roadmap derived from the project spec (pages 10–11). Items are grouped by spec
area and marked with priority. Bracketed tags map back to the spec:
**[Core]** = required deliverable, **[Bonus]** = stretch goal, **[Eval]** = validation.

> **Note on sensing:** all camera capture and signal classification happens in
> the **frontend** (MediaPipe JS). The backend only consumes pre-classified
> labels (`affect`, `gesture_tag`, `gaze_bucket`).

### Dataset

- [ ] **[Core]** Add **heterogeneous** memory types per persona — currently only
      autobiographical narratives exist.
  - [ ] Add a set of synthetic social-media posts per persona (voice-matched)
  - [ ] Add a set of synthetic past communication logs per persona
  - [ ] Regenerate the synthesis script to produce both, then rebuild embeddings
  - [ ] Make ingestion type-aware so the retriever knows which chunk-type a hit came from
- [ ] **[Core]** Document the dataset schema so it is reusable by the evaluation harness.

### Multimodal Sensing (frontend)

- [ ] **[Core]** Detect **head-nod / sharp tilt as dissatisfaction**, distinct
      from a generic frustrated affect read.
  - [ ] Send a `dissatisfaction_signal` to the backend alongside the existing labels
  - [ ] When the signal fires, branch the planner to a **"Turnaround Option"** —
        a clarification candidate ("Did you mean X or Y?") instead of a plain answer
- [ ] **[Bonus]** Add **vocalisation capture** (Web Speech API) and a
      **conflict-resolution** step that compares the spoken intent against the
      air-written intent, sending a single `resolved_intent` to the backend.
- [ ] **[Polish]** Tighten the **thumbs-up boost** — today it only annotates the
      prompt. The retriever should also bias affirmative-leaning candidates when
      a thumbs-up is present.

### Agentic Intent Decomposition

> **Current state:** intent routing is **keyword-based**, not LLM-based.
> The original LLM-driven router (Pydantic-validated JSON output) was
> dropped because `gemma4:31b-cloud` consistently emitted the wrong JSON
> shape and got truncated by `max_tokens`, triggering 3 retries + a
> hard-fallback on every turn — adding ~30s of dead latency before the
> generation call. The keyword router (~5 buckets matched against
> hardcoded word lists in `intent.py`) handles the demo personas
> reliably and adds ~0ms per turn.
>
> **Trade-off:** the router is limited to the 5 hardcoded buckets
> (`family`, `medical`, `hobbies`, `daily_routine`, `social`) and can't
> distinguish `OPEN_DOMAIN` from `PERSONAL` queries. Acceptable today
> because all current personas only have personal memories.

- [ ] **[Core]** Make Personal / Contextual / Open-domain routing actually hit
      **different retrieval pools** — today all sub-queries fall back to the same
      vector index. Requires re-introducing some form of intent classification
      (likely a constrained-output LLM call once `response_format=json_schema`
      is supported on Ollama Cloud, or a tiny local classifier).
- [ ] **[Perf]** When/if we re-add LLM intent: cache the schema prompt,
      use a smaller routing model, and parallelise sub-query retrieval.

### Retrieval

- [ ] **[Bonus]** Persist **bucket priors** per user across conversations
      (currently per-session only).
- [ ] **[Bonus]** Extend the **latency-optimised fallback** beyond a single
      LLM-tier switch:
  - [ ] Return a cached canned response when end-to-end latency blows the budget
  - [ ] Use the spec's **< 6s end-to-end** target instead of the current 3.5s threshold
- [ ] **[Scale]** When per-user memory grows past ~100k chunks, swap the
      torch-tensor matmul search for `hnswlib` (a ~2 MB approximate-NN library);
      reintroduce a cross-encoder reranker once `top_k > ~30`.

### Training-Free Response Generation

- [ ] **[Core]** Return **multiple candidate responses** from the API so the
      user can pick one (today the endpoint returns a single string).
- [ ] **[Bonus]** On user selection, upsert the `(query, selected_response)` pair
      into a small "accepted-pairs" index and consult it as a high-prior shortcut
      on the next turn — the spec's lightweight retrieval-index update.

### Evaluation & Validation

- [ ] **[Eval]** **Factual Faithfulness** — NLI-based groundedness metric over
      (retrieved evidence, generated response) pairs, reported as a hallucination
      rate on a held-out set of partner-style queries per persona.
- [ ] **[Eval]** **Communication Efficiency** — p50 / p95 end-to-end latency
      across all three LLM tiers, with a pass/fail gate at the spec target of
      **< 6s p95**.
- [ ] **[Eval]** **Perceived Authenticity** — generate paired (persona, query,
      response) samples and a 5-point Likert rating sheet for the live in-class eval.
- [ ] **[Eval]** **Multimodal Alignment** — synthetic (gesture, query) scenarios
      checked against expected response traits (e.g. thumbs-up ⇒ affirmative
      lexicon present), reported as alignment accuracy.

### Polish

- [ ] **[Polish]** Move the hard-coded affect→tone and persona-override dicts
      into a single YAML so tone-shaping can be tuned without touching code.
- [x] **[Polish]** Delete the unused `backend/sensing/` Python modules now that
      sensing lives entirely in the frontend. *(Done — only `labels.py` remains.)*
>>>>>>> Stashed changes

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

- [ ] **[Core]** Personal / Contextual / Open-domain all hit the same FAISS index right now. Make them actually go different places — open-domain → web search (or stub), contextual → session memory
- [ ] intent node is slow. Cache the prompt, use a tiny model for routing, parallelise the sub-queries

### Retrieval

- [ ] **[Bonus]** Bucket priors only live for the session. Persist them per user
- [ ] **[Bonus]** Latency fallback only switches LLM tier. Add more steps:
  - drop reranker if retrieval is slow
  - return a canned response if we blow the budget entirely
  - threshold is 3.5s, spec says 6s — pick one

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
- [ ] delete `backend/sensing/` (dead code, sensing is in frontend)

---

## Team

- **Akash Kolte** — akashjag@buffalo.edu
- **Shwetangi** — shwetang@buffalo.edu

University at Buffalo, SUNY

---

## License

All rights reserved. See the [LICENSE](LICENSE) file for details.
