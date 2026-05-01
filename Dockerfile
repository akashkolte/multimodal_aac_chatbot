# Stage 1 — build React frontend
FROM node:22-alpine AS frontend-build
WORKDIR /app/frontend
RUN npm install -g pnpm
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
RUN pnpm build

# Stage 2 — Python backend + static frontend
FROM python:3.11-slim
WORKDIR /app

# CPU-only torch first — keeps image ~1.5 GB smaller than default GPU wheel
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY data/ ./data/
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

RUN mkdir -p logs

EXPOSE 8000
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
