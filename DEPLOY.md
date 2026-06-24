# Deploying BFSI Policy Assistant

Fully open-source stack:
- **LLM:** open-weight model via the **Hugging Face Inference API** (no proprietary API).
- **Embeddings:** open-source `sentence-transformers`, in-process on CPU.
- **Host:** **Hugging Face Spaces** (Docker SDK), free tier.

The app's real server is a **Flask** app (the Streamlit page is only a local
launcher), served by **gunicorn** inside the container on port 7860.

> Note: Vercel cannot host this. It needs an always-on server, not short-lived
> serverless functions.

## Steps (Hugging Face Spaces, Docker SDK)
1. Create a Space at https://huggingface.co/new-space → **SDK: Docker**, hardware **CPU basic** (free).
2. Push this repo's contents to the Space (it includes a `Dockerfile`).
3. Add this YAML to the top of the Space's `README.md`:
   ```yaml
   ---
   title: BFSI Policy Assistant
   sdk: docker
   app_port: 7860
   ---
   ```
4. In the Space: **Settings → Variables and secrets**, add a secret:
   - `HF_TOKEN` = your token from https://huggingface.co/settings/tokens
   - (optional) `FLASK_SECRET_KEY` = a long random string
5. The Space builds and serves the app. `/` redirects to the company portal.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env   # add your HF_TOKEN
python app.py          # local Streamlit + Flask launcher
```

## Limitations
- Uploaded docs + the Chroma vector DB live on the container disk and **reset on
  rebuild / restart**. For persistence, attach HF persistent storage or move the
  vector store to a hosted DB.
- Keep a single gunicorn worker: retrievers and session state are held in memory.
