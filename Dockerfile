# Hugging Face Space (Docker SDK). HF serves the app on port 7860.
FROM python:3.11-slim

WORKDIR /app

# System deps occasionally needed by chromadb / sentence-transformers builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-cache the embedding model so first request is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

ENV PORT=7860
EXPOSE 7860

# Single worker: retrievers and session state are held in-process.
CMD ["gunicorn", "app:flask_app", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "180"]
