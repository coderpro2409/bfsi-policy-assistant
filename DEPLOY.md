# Deploying BFSI Policy Assistant

This app's real server is a **Flask** app (the Streamlit page is only a local
launcher). It is deployed as an always-on web service on **Render** (free tier).

> Note: Vercel cannot host this. It needs an always-on server, not short-lived
> serverless functions.

## What changed for cloud deploy
- LLM now uses **OpenRouter** (hosted) instead of local Ollama.
- Embeddings run **in-process on CPU** via sentence-transformers (no API key).
- Served by **gunicorn** (`app:flask_app`), binding to `$PORT`.

## Steps (Render)
1. Push this repo to GitHub.
2. Go to https://render.com → **New** → **Web Service** → connect this repo.
   Render auto-detects `render.yaml`. (Or set it manually:
   build `pip install -r requirements.txt`,
   start `gunicorn app:flask_app --bind 0.0.0.0:$PORT --workers 1 --timeout 180`.)
3. In **Environment**, add a secret:
   - `OPENROUTER_API_KEY` = your key from https://openrouter.ai/keys
4. Deploy. Open the service URL (`/` redirects to the company portal).

## Limitations
- Uploaded docs + the Chroma vector DB live on local disk and **reset on
  redeploy / when the free instance sleeps**. For persistence, attach a Render
  paid disk or move the vector store to a hosted DB.
- Keep `--workers 1`: retrievers and session state are held in memory per process.
