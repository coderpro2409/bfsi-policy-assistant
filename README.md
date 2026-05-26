# BFSI Policy Assistant

A **RAG (Retrieval-Augmented Generation) chatbot** for **Banking, Financial
Services & Insurance (BFSI) policy documents**.

Upload your PDFs / DOCX / TXT, and ask natural-language questions about
them. The pipeline uses **local embeddings** + a **local vector store**
for retrieval, and a **cloud-hosted Mistral 7B** model (via OpenRouter's
free tier) for answer generation.

Built with:
- **[Streamlit](https://streamlit.io/)** — UI
- **[LangChain](https://www.langchain.com/)** — RAG orchestration
- **[Ollama](https://ollama.com/)** — local embeddings (`llama3`)
- **[Chroma](https://www.trychroma.com/)** — local persistent vector store
- **[OpenRouter](https://openrouter.ai/)** — cloud LLM gateway (Mistral 7B)

---

## Features

- **Multi-format ingestion** — PDF, DOCX, and TXT documents
- **Persistent vector store** — Chroma writes to disk (`./chroma_db_storage/`)
  so your index survives restarts; no re-embedding on every launch
- **Dynamic query expansion** — generates multiple query variations from
  the user's question to improve recall
- **Hybrid retrieval** — combines plain similarity search with **MMR**
  (Maximal Marginal Relevance) for diverse, non-redundant context
- **Heading-aware formatting** — extracts section / policy / clause headings
  so answers cite "**Section 4.2**" instead of "chunk #7"
- **Source attribution** — every answer shows which document(s) it drew
  from, with page numbers when available
- **Free-tier friendly** — uses OpenRouter's free Mistral 7B endpoint

---

## Privacy note

Unlike a fully local RAG stack, the **final answer step calls
OpenRouter's API**. That means the retrieved document chunks (the
context the model sees) are sent to OpenRouter over the network.

- Embeddings and the vector store are **local**.
- The raw uploaded files **never leave your machine**.
- But **selected chunks** of those files **do** get sent to OpenRouter
  every time the model answers a question.

If your documents are sensitive enough that even chunks shouldn't leave
your machine, swap `ChatOpenAI` for `ChatOllama` and use a local model
instead.

---

## Requirements

- **Python 3.9+**
- **[Ollama](https://ollama.com/download)** installed and running (used
  for local embeddings).
  ```bash
  ollama pull llama3
  ```
- An **[OpenRouter](https://openrouter.ai/keys)** account and API key.
  Mistral 7B is free-tier; sign up and create a key at
  https://openrouter.ai/keys.
- ~1–3 GB of free disk for the Chroma collection (depends on document
  size).

---

## Setup

```bash
# 1. Clone
git clone https://github.com/coderpro2409/bfsi-policy-assistant.git
cd bfsi-policy-assistant

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Make sure Ollama is running and has the embedding model
ollama pull llama3
ollama serve                       # if not already running

# 5. Configure your OpenRouter API key
cp .env.example .env
# then open .env and paste your OPENROUTER_API_KEY
```

### Loading the `.env` file

The script reads from real environment variables, not the `.env` file
directly. Easiest way to load them for one run:

```bash
set -a; source .env; set +a
streamlit run app.py
```

(Or use `direnv` / `python-dotenv` if you prefer.)

---

## Usage

```bash
streamlit run app.py
```

Then in the browser:

1. **Upload one or more policy documents** (PDF / DOCX / TXT) from the
   sidebar.
2. Click **"Process Documents"** — this chunks them, computes embeddings
   locally via Ollama, and stores them in the Chroma collection.
3. **Ask a question** in the chat input. The app retrieves relevant
   chunks locally, then sends `{question + chunks}` to Mistral 7B via
   OpenRouter for the final answer.

Each answer shows the response, the retrieved sections, and the source
document + page number for each citation.

---

## How it works

```
                ┌───────────────────────────────────────────────────┐
                │                    app.py (Streamlit UI)          │
                └───────────────────────────────────────────────────┘
                                       │
        upload PDFs/DOCX/TXT            │            user question
                  │                     │                  │
                  ▼                     │                  ▼
       ┌────────────────────┐           │      ┌─────────────────────┐
       │ Document loaders   │           │      │ DynamicQueryExpander│
       │ (LangChain)        │           │      │  → query variations │
       └─────────┬──────────┘           │      └──────────┬──────────┘
                 │ chunks                                  │
                 ▼                                         ▼
       ┌────────────────────┐               ┌────────────────────────┐
       │ Ollama embeddings  │               │ Dynamic retriever      │
       │ (llama3) — LOCAL   │──────────────▶│ similarity + MMR       │
       └─────────┬──────────┘               └──────────┬─────────────┘
                 │                                     │ top-k chunks
                 ▼                                     ▼
       ┌─────────────────────────────────────────────────────┐
       │              Chroma vector store                    │
       │              (./chroma_db_storage/)                 │
       └─────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
                                       ┌──────────────────────────────┐
                                       │ OpenRouter — Mistral 7B      │
                                       │  ChatOpenAI (CLOUD CALL)     │
                                       │  → final answer              │
                                       └──────────────────────────────┘
```

---

## Project structure

```
bfsi-policy-assistant/
├── app.py             # Streamlit app + RAG pipeline (single file)
├── requirements.txt   # Python dependencies
├── .env.example       # Template for OPENROUTER_API_KEY
├── .gitignore         # Ignores .env, .venv, chroma_db_storage/, uploaded docs
├── LICENSE            # MIT
└── README.md
```

At runtime, the app creates:

```
├── chroma_db_storage/    # Persisted vector store (git-ignored)
```

---

## Notes & limitations

- **First indexing pass is slow.** Embedding a large policy document on
  CPU via Ollama can take several minutes. Subsequent launches reuse the
  Chroma collection.
- **OpenRouter rate limits apply** on the free tier. If you hit a
  `429 Too Many Requests`, wait a few seconds and retry — or upgrade.
- **No multi-user isolation.** Everyone hitting the same Streamlit
  instance shares the same Chroma collection. Fine for personal use; not
  suitable for multi-tenant deployment as-is.
- **Document files are git-ignored by default.** Use `git add -f file.pdf`
  if you want to ship a sample document with the repo.

---

## License

[MIT](./LICENSE)
