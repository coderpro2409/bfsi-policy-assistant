# Deploying BFSI Policy Assistant

Fully self-hosted, open-source stack:
- **LLM + embeddings:** **Ollama** (open-weight models, no external service).
- **App:** Flask served by gunicorn (the Streamlit page is only a local launcher).

> Important: this needs a **VM with enough RAM** (Ollama runs the model in
> memory: ~4GB for a 3B model, ~8GB for `llama3` 8B). It cannot run on free
> serverless hosts (Vercel, Render free, etc.), which have no GPU, little RAM,
> and no persistent process. A free **Oracle Cloud Always Free** ARM instance
> (up to 24GB RAM) is a good fit.

## Deploy on a VM with Docker (recommended)
```bash
# on the VM, in this repo directory
docker compose up -d --build
docker compose exec ollama ollama pull llama3
docker compose exec ollama ollama pull nomic-embed-text
```
Open `http://<vm-ip>:8000` (open port 8000 in the VM firewall). `/` redirects
to the company portal.

To use a smaller, faster model on CPU:
```bash
docker compose exec ollama ollama pull llama3.2:3b
# then set LLM_MODEL=llama3.2:3b (e.g. in a .env file next to compose) and:
docker compose up -d
```

## Run locally
```bash
ollama serve &
ollama pull llama3 && ollama pull nomic-embed-text
pip install -r requirements.txt
python app.py            # local Streamlit + Flask launcher
```

## Limitations
- Uploaded docs + the Chroma vector DB live on the app container's disk and
  **reset on rebuild**. Add a Docker volume for persistence if needed.
- Keep a single gunicorn worker: retrievers and session state are in memory.
