# ── Stage 1: build the React frontend ────────────────────────────────────────
FROM node:22-slim AS frontend

WORKDIR /app/frontend

# pnpm via corepack (ships with Node 22)
RUN corepack enable

COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

COPY frontend/ ./
RUN pnpm build

# ── Stage 2: Python runtime ──────────────────────────────────────────────────
FROM python:3.12-slim

# HF_HOME points at a writable cache dir for transformers/sentence-transformers.
# On HF Spaces the default $HOME is read-only at runtime, so we explicitly
# steer the model cache somewhere writable.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf_cache \
    XDG_CACHE_HOME=/tmp/.cache

WORKDIR /app

# System deps for torch + sentence-transformers (most are already in slim).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps via a Docker-specific requirements file that pins torch
# to the CPU-only wheel index. The base requirements.txt stays platform-neutral
# so local conda dev (./setup.sh) keeps using whatever torch flavor your OS
# wants (MPS on macOS, CUDA on Linux+GPU); this image deliberately uses CPU
# only because HF Spaces' free CPU instance can't use CUDA anyway.
COPY requirements.txt requirements-docker.txt ./
RUN pip install --upgrade pip \
    && pip install --retries 5 --timeout 120 -r requirements-docker.txt

# Copy the backend + persona source data.
COPY backend/ ./backend/
COPY data/memories/ ./data/memories/
COPY data/users.json ./data/users.json
COPY data/generate_users.py ./data/generate_users.py

# Build per-user vector indexes inside the image (downloads BGE on first run).
# This bakes the indexes into the image so first-request latency is just the
# model warm-up, not a fresh BGE encode of every persona.
RUN python -m backend.retrieval.vector_store

# Pull the built static frontend from stage 1.
COPY --from=frontend /app/frontend/dist ./frontend/dist

# Pre-create writable directories. HF Spaces filesystem is read-only outside
# /tmp at runtime, so logs default to /tmp; locally you can override LOGS_DIR
# via env to anything mounted/writable.
RUN mkdir -p /tmp/logs /tmp/hf_cache /tmp/.cache /tmp/pick_index \
    && chmod -R 777 /tmp/logs /tmp/hf_cache /tmp/.cache /tmp/pick_index
ENV LOGS_DIR=/tmp/logs

# HF Spaces expects 7860 by default; respects $PORT for local docker run.
ENV PORT=7860
EXPOSE 7860

# sh -c expands $PORT at runtime so the same image runs both on HF (port 7860,
# unset PORT or PORT=7860) and locally (e.g. `docker run -e PORT=8000 ...`).
CMD sh -c "uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-7860}"
