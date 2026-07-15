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

## Deploying to Oracle Cloud (VM.Standard.A1.Flex)

### First-time setup

```bash
# 1. SSH into the VM
ssh -i ~/Downloads/ssh-key.key ubuntu@<YOUR_VM_PUBLIC_IP>

# 2. Install Docker
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker ubuntu
newgrp docker

# 3. Clone the repo
git clone https://github.com/akashkolte/multimodal_aac_chatbot.git
cd multimodal_aac_chatbot

# 4. Create your .env file (fill in your API keys)
nano .env

# 5. Open port in Ubuntu firewall
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 7860 -j ACCEPT
sudo netfilter-persistent save

# 6. If Docker has DNS issues, restart the daemon
sudo systemctl daemon-reload
sudo systemctl restart docker

# 7. Build and run
docker build -t aac-chatbot .
docker run -d --name chatbot --restart unless-stopped \
  -p 7860:7860 \
  --env-file ~/multimodal_aac_chatbot/.env \
  aac-chatbot
```

> **HTTPS & Camera:** Camera access requires HTTPS. Use [Caddy](https://caddyserver.com/) as a reverse proxy with your domain to get automatic SSL. Point it to `localhost:7860`.

### Re-deploying after code changes

```bash
ssh -i ~/Downloads/ssh-key.key ubuntu@<YOUR_VM_PUBLIC_IP>
cd ~/multimodal_aac_chatbot
git pull
docker stop chatbot && docker rm chatbot
docker build -t aac-chatbot .
docker run -d --name chatbot --restart unless-stopped \
  -p 7860:7860 \
  --env-file ~/multimodal_aac_chatbot/.env \
  aac-chatbot
```

### Useful commands

```bash
docker logs -f chatbot        # watch live logs
docker ps                     # check container is running
docker exec -it chatbot bash  # shell into the container
```

---

## License

All rights reserved. See the [LICENSE](LICENSE) file for details.
