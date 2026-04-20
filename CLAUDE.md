# Multimodal AAC Chatbot — Project Guide

## What This Project Does

An AI chatbot that **speaks as an AAC user**, not to them. Given one of 14
personas — nine anchored in real memoirs and five in canonical fiction —
it fuses real-time multimodal non-verbal signals with personal memory
retrieval to generate responses in that person's authentic voice. Orchestrated
as a **plain Python function chain** across five layers, with two conditional
branches.

---

## Architecture

```
frontend/                         React + Vite + TypeScript
  src/hooks/useSensing.ts         MediaPipe JS — affect, gesture, gaze, air-writing (browser-side)
  src/components/ChatPanel.tsx    Chat UI → POST /chat with sensing labels

backend/                          Python (conda env: aac-chatbot)
  main.py                         CLI entry point
  api/main.py                     FastAPI REST API
  pipeline/graph.py               run_pipeline() — plain function chain with 2 conditional branches
    pipeline/nodes/intent.py        L2 — LLM + Pydantic intent routing
    pipeline/nodes/retrieval.py     L3 — BGE embeddings + torch tensor cosine search (fast / full)
    pipeline/nodes/planner.py       L4 — expression-conditioned generation
    pipeline/nodes/feedback.py      L5 — JSONL turn logging + Bayesian bucket priors
  sensing/labels.py               GESTURE_TO_TAG label map (sensing itself runs in browser)
  retrieval/                      BGE embeddings (torch), Bayesian bucket priors
  generation/                     Two-tier LLM client (primary / fallback, both Ollama Cloud)
  guardrails/                     Input + output safety checks
  config/                         Pydantic BaseSettings — all config in one place

data/                             Shared data (personas, vector indexes)
logs/                             Per-turn JSONL logs (gitignored)
```

## Key Design Decisions

- **Plain function chain** orchestrates the pipeline (`run_pipeline` in
  `backend/pipeline/graph.py`): intent → retrieval → planner → feedback,
  with two conditional branches (affect picks fast/full retrieval; cumulative
  latency picks primary/fallback LLM). No LangGraph / LangChain dependency.
- **BGE-small-en-v1.5** for embeddings (beats MiniLM on MTEB at same speed)
- **Torch tensor matmul** for vector search on the embedder's device
  (mps → cuda → cpu). No FAISS, no separate index format. Stored as
  `vectors.pt` per user. Headroom is ~100k vectors before approximate
  search (`hnswlib`) becomes worthwhile.
- **No reranker** — cosine score from BGE-small carries the ranking signal
  at current scales. Revisit when per-query `top_k` grows past ~30.
- **Two-tier Ollama Cloud LLM**: `primary` → `fallback` (when cumulative
  latency exceeds `FALLBACK_LATENCY_THRESHOLD`). Both tiers hit Ollama
  Cloud over the OpenAI-compatible endpoint. Models default to
  `gemma4:31b-cloud`; swap one when a larger cloud model is provisioned.
- **Pydantic-validated** LLM routing output — `intent.py` retries on schema
  failures (3 attempts) before falling back to a default route
- **Expression-conditioned response shaping** — affect steers tone, retrieval depth,
  and candidate ranking (not just metadata annotation)
- **Bayesian bucket priors** — session-level P(bucket) updated after each accepted turn
- **Per-turn JSONL logging** — one line per turn appended to
  `logs/turns.jsonl` (no MLflow). Query ad-hoc with DuckDB if needed.
- **Browser-side sensing** — MediaPipe JS runs in React frontend, only classified
  labels (affect, gesture, gaze bucket) are sent to the backend API

---

## Personas

Fourteen personas shipped. Real-memoir-anchored:

| ID | Name | Condition | Access |
|----|------|-----------|--------|
| `stephen_hawking` | Stephen Hawking | ALS (advanced) | Cheek-twitch + ACAT predictive speech |
| `jean_dominique_bauby` | Jean-Dominique Bauby | Locked-in syndrome | Alphabet-blink with amanuensis |
| `michael_j_fox` | Michael J. Fox | Parkinson's | Voice + adaptive keyboard + dictation |
| `gabby_giffords` | Gabby Giffords | Aphasia + right hemiparesis (post-TBI) | Left-hand typing + speech-to-text |
| `jason_becker` | Jason Becker | ALS (fully locked-in) | Eye-gaze + father's letter-code board |
| `tito_mukhopadhyay` | Tito Mukhopadhyay | Non-verbal autism | Letterboard + pencil |
| `christopher_reeve` | Christopher Reeve | C1–C2 spinal cord injury | Dictation to assistants; sip-and-puff |
| `christy_brown` | Christy Brown | Cerebral palsy (spastic quadriplegia) | Left foot typing / writing |
| `wendy_mitchell` | Wendy Mitchell | Early-onset dementia | Laptop/phone typing + "brain-book" |

Canonical fiction:

| ID | Name | Condition | Access |
|----|------|-----------|--------|
| `abed_nadir` | Abed Nadir (*Community*) | Autism (coded); occasional selective mutism | Mostly verbal; text when overloaded |
| `allie_calhoun` | Allie Hamilton Calhoun (*The Notebook*) | Late-stage Alzheimer's | Verbal when lucid; yes/no otherwise |
| `forrest_gump` | Forrest Gump | Intellectual disability (IQ ~75) | Verbal primarily |
| `raymond_babbitt` | Raymond Babbitt (*Rain Man*) | Savant autism | Verbal when calm + visual schedules |
| `walter_jr_white` | Walter "Flynn" White Jr. (*Breaking Bad*) | Cerebral palsy | Verbal + smartphone typing |

~25 bucketed memory chunks per persona (`family` / `medical` / `hobbies` / `daily_routine` / `social`; buckets tuned per-persona). A short-form voice push-to-talk mic surfaces only for personas whose modelled access method is verbal — see `VOICE_CAPABLE_PERSONAS` in [frontend/src/lib/voiceEligibility.ts](frontend/src/lib/voiceEligibility.ts).

---

## How to Run

```bash
# One-time setup
bash setup.sh

# CLI
python -m backend.main --debug

# Full stack
uvicorn backend.api.main:app --reload    # FastAPI on :8000
pnpm --dir frontend dev                  # React on :7550
```

---

## Configuration

All config lives in [backend/config/settings.py](backend/config/settings.py) as Pydantic `BaseSettings`.
Copy `.env.example` → `.env` and set:

- `ACTIVE_LLM_TIER` — `primary` | `fallback`
- `PRIMARY_MODEL` / `FALLBACK_MODEL` — Ollama Cloud model identifiers
  (e.g. `gemma4:31b-cloud`)
- `LOGS_DIR` — where per-turn JSONL logs are written (default: `logs/`)

---

## Data Files

| Path | Purpose |
|------|---------|
| `data/users.json` | Flat user index (id, name, condition, style) |
| `data/memories/<uid>.json` | Full persona JSON with bucketed memories |
| `data/vector_store/<uid>/` | `vectors.pt` + `meta.json` — **rebuild after any persona edit** |
| `data/generate_users.py` | Regenerates memories + users.json |

---

## Code Style

- **Keep comments to a minimum.** Only comment what isn't obvious from the
  code. No file headers explaining what a module does (the name and code
  show that). No section divider banners (`# ── Foo ──`). No restating
  what the next line does. Prefer one-line comments when needed.
- **Skip `from __future__ import annotations`.** The project is Python 3.10+
  and uses native `X | None` / `list[dict]` syntax — the import adds nothing.

## Development Notes

- **NEVER use local Ollama models** (e.g. `qwen3:8b`, `gemma3:1b`) — this machine
  is not powerful enough and will break. Always use cloud-backed models like
  `gemma4:31b-cloud` via Ollama Cloud.
- **Adding a persona**: add a memory JSON under `data/memories/<uid>.json` and
  a matching entry in `data/users.json` (or regenerate both via
  `data/generate_users.py` if present), then
  `python -m backend.retrieval.vector_store` to rebuild indexes. If the
  persona's modelled access method includes live speech, also add their `id`
  to `VOICE_CAPABLE_PERSONAS` in `frontend/src/lib/voiceEligibility.ts` so
  the mic button surfaces.
- **Changing LLM**: set `ACTIVE_LLM_TIER` in `.env` — no code changes needed
- **Extending sensing**: sensing runs in the React frontend
  (`frontend/src/hooks/useSensing.ts`); to add a new signal, classify it
  there and add a label field to `PipelineState` in
  `backend/pipeline/state.py`. Keep purely-data label maps in
  `backend/sensing/labels.py`.
- **Guardrail tuning**: edit signal lists in `backend/guardrails/checks.py`
- **Affect → generation mapping**: `_AFFECT_CONFIG` in `backend/pipeline/nodes/intent.py`
  and `_PERSONA_TONE_OVERRIDES` in `backend/pipeline/nodes/planner.py`
- Vector indexes in `data/vector_store/` are gitignored — rebuilt from source JSONs
  via `python -m backend.retrieval.vector_store`
- Frontend uses pnpm, Node 22+
