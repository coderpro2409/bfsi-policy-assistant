# App container (Flask + gunicorn). Talks to a separate Ollama container
# via OLLAMA_BASE_URL (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# Single worker: retrievers and session state are held in-process.
CMD ["sh", "-c", "gunicorn app:flask_app --bind 0.0.0.0:${PORT:-8000} --workers 1 --timeout 180"]
