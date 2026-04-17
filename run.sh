#!/usr/bin/env bash
set -euo pipefail

export PYTHONWARNINGS="ignore::UserWarning:multiprocessing.resource_tracker"

CONDA_ENV="aac-chatbot"

# Activate conda env
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Run setup.sh first." >&2
  exit 1
fi
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# If any args were passed (e.g. --debug, --user mia_chen), run the CLI
# instead of the full stack and forward them verbatim.
if [ "$#" -gt 0 ]; then
  exec python -m backend.main "$@"
fi

PIDS=()

cleanup() {
  echo ""
  echo "Shutting down..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null
  done
  # Wait for them to exit, suppress all output
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null
  done
  exit 0
}

trap cleanup INT TERM

# Use Node 22 if available (Vite 8 requires Node 20.19+ or 22.12+)
if [ -x /opt/homebrew/opt/node@22/bin/node ]; then
  export PATH="/opt/homebrew/opt/node@22/bin:/opt/homebrew/bin:$PATH"
fi

# Start Ollama if not already running
if command -v ollama >/dev/null 2>&1 && ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama..."
  ollama serve >/dev/null 2>&1 &
  PIDS+=($!)
  sleep 2
fi

echo "Starting FastAPI backend on :8000..."
uvicorn backend.api.main:app --reload --port 8000 2>&1 &
PIDS+=($!)

# Wait for backend to be reachable before starting frontend
echo "Waiting for backend..."
until curl -s http://localhost:8000/health >/dev/null 2>&1; do
  sleep 1
done
echo "Backend ready."

echo "Starting React frontend on :7550..."
pnpm --dir frontend dev 2>&1 &
PIDS+=($!)

echo "All services running. Ctrl+C to stop."
wait
