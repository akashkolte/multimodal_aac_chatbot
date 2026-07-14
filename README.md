# Multimodal AAC Chatbot

A chatbot that **speaks as an AAC user, not to them.** Select a persona and the conversation partner talks to them — the bot replies in that person's voice using their memories, adjusting responses based on webcam input: facial expression, hand gestures, gaze direction, and letters traced in the air.

Built as a training-free agentic RAG pipeline with a FastAPI backend and a React + Vite frontend.

---

## Team

| Name | Email |
|---|---|
| **Akash Kolte** | akashjag@buffalo.edu |
| **Shwetangi** | shwetang@buffalo.edu |

*University at Buffalo, SUNY*

---

## Running the Backend

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.com) **or** an NVIDIA NIM API key

### Setup (first time only)

```bash
# 1. Create and activate a virtual environment (from project root)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment variables
cp .env.example .env
# Edit .env — set your API key and model name

# 4. Build vector indexes (downloads embedder ~130MB on first run)
python -m backend.retrieval.vector_store
```

### Run

```bash
source .venv/bin/activate
uvicorn backend.api.main:app --reload --port 8000
```

Backend runs at **http://localhost:8000** · Health check: `http://localhost:8000/health`

---

## Running the Frontend

### Prerequisites
- Node.js 20.19+ or 22.12+
- [pnpm](https://pnpm.io) (`npm i -g pnpm`)

### Setup (first time only)

```bash
cd frontend
pnpm install
```

### Run

```bash
pnpm run dev
```

Frontend runs at **http://localhost:5173** (or the port shown in the terminal).

---

## Running Both Together

```bash
# Terminal 1 — backend
source .venv/bin/activate
uvicorn backend.api.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && pnpm run dev
```

Or use the convenience script (requires conda):

```bash
./run.sh
```

---

## License

All rights reserved. See the [LICENSE](LICENSE) file for details.
