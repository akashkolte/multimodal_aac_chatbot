#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="aac-chatbot"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"

info()  { printf "\033[1;34m==> %s\033[0m\n" "$1"; }
ok()    { printf "\033[1;32m==> %s\033[0m\n" "$1"; }
warn()  { printf "\033[1;33m==> %s\033[0m\n" "$1"; }
fail()  { printf "\033[1;31mERROR: %s\033[0m\n" "$1"; exit 1; }

command -v conda >/dev/null 2>&1 || fail "conda not found. Install Miniconda/Anaconda first."

if conda info --envs | grep -q "^${CONDA_ENV} "; then
  info "Conda env '$CONDA_ENV' already exists — reusing it"
else
  info "Creating conda env '$CONDA_ENV' (Python 3.12)..."
  conda create -n "$CONDA_ENV" python=3.12 -y --quiet
  ok "Conda env created"
fi

# Activate inside this script
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

info "Installing Python dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
ok "Dependencies installed"

if [ -f "$ENV_FILE" ]; then
  warn ".env already exists — skipping copy (review $ENV_EXAMPLE for new vars)"
else
  info "Copying $ENV_EXAMPLE → $ENV_FILE..."
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  ok ".env created — edit it to configure Ollama Cloud model names"
fi

info "Building vector indexes (downloads BGE-small embedder on first run)..."
python -m backend.retrieval.vector_store
ok "Vector indexes built in data/faiss_store/"

# Ollama: tiers point at Ollama Cloud — no local pull needed. Just check the
# daemon is reachable so the OpenAI-compatible proxy works.
if ! command -v ollama >/dev/null 2>&1; then
  warn "Ollama not installed — install it from https://ollama.com then re-run this script"
fi

if command -v pnpm >/dev/null 2>&1; then
  info "Installing frontend dependencies..."
  pnpm --dir frontend install --silent
  ok "Frontend dependencies installed"
else
  warn "pnpm not found — install it (npm i -g pnpm) then run: pnpm --dir frontend install"
fi

echo ""
ok "Setup complete!"
echo ""
echo "  Activate the environment:"
echo "    conda activate $CONDA_ENV"
echo ""
echo "  Run the CLI:"
echo "    python -m backend.main --debug"
echo ""
echo "  Or start the full stack:"
echo "    uvicorn backend.api.main:app --reload    # FastAPI on :8000"
echo "    pnpm --dir frontend dev                  # React on :7550"
echo ""
