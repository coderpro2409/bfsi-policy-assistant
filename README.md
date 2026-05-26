# BFSI Policy Assistant

A local, fully offline **RAG (Retrieval-Augmented Generation) chatbot** for
**Banking, Financial Services & Insurance (BFSI) policy documents**.

Upload your PDFs / DOCX / TXT, and ask natural-language questions about them.
Everything — the embeddings, the vector store, and the LLM — runs on your
own machine. No cloud, no API keys, no document upload to a third party.

Built with:
- **[Streamlit](https://streamlit.io/)** — UI
- **[LangChain](https://www.langchain.com/)** — RAG orchestration
- **[Ollama](https://ollama.com/)** — local LLM (`llama3`) + embeddings
- **[Qdrant](https://qdrant.tech/)** — local persistent vector store

---

## Features

- **Multi-format ingestion** — PDF, DOCX, and TXT documents
- **Persistent vector store** — Qdrant writes to disk (`./qdrant_storage/`)
  so your index survives restarts; no re-embedding on every launch
- **Dynamic query expansion** — generates multiple query variations from
  the user's question (key-phrase extraction + term pairs) to improve recall
- **Hybrid retrieval** — combines plain similarity search with **MMR**
  (Maximal Marginal Relevance) for diverse, non-redundant context
- **Heading-aware formatting** — extracts section / policy / clause headings
  from chunks so answers can cite "**Section 4.2**" instead of "chunk #7"
- **Source attribution** — every answer shows which document(s) it drew
  from, with page numbers when available
- **Zero data leaves your machine** — no external API calls

---

## Requirements

- **Python 3.9+**
- **[Ollama](https://ollama.com/download)** installed and running
- A local model pulled (default is `llama3`):
  ```bash
  ollama pull llama3
  ```
  > The same model is used for both the LLM **and** embeddings. If you'd
  > rather use a dedicated embedding model (e.g. `nomic-embed-text`), edit
  > the `MODEL_NAME` constant near the top of `app.py`.
- ~2–4 GB of free disk for the Qdrant collection (depends on document size)

---

## Setup

```bash
# 1. Clone
git clone https://github.com/coderpro2409/bfsi-policy-assistant.git
cd bfsi-policy-assistant

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Make sure Ollama is running and the model is available
ollama pull llama3
ollama serve                       # if it isn't already running
```

---

## Usage

```bash
streamlit run app.py
```

Then in the browser:

1. **Upload one or more policy documents** (PDF / DOCX / TXT) from the
   sidebar.
2. Click **"Process Documents"** — this chunks them, computes embeddings
   via Ollama, and stores them in the local Qdrant collection.
3. **Ask a question** in the chat input. The app retrieves relevant chunks
   and asks the local `llama3` model to answer using only that context.

Each answer shows:
- The generated response
- The retrieved sections it relied on
- The source document and page number for each citation

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
                 │ chunks                            │
                 ▼                                   ▼
       ┌────────────────────┐               ┌────────────────────┐
       │ Ollama embeddings  │               │ DynamicRetriever   │
       │ (llama3)           │──────────────▶│ similarity + MMR   │
       └─────────┬──────────┘               └──────────┬─────────┘
                 │                                     │ top-k chunks
                 ▼                                     ▼
       ┌─────────────────────────────────────────────────────┐
       │              Qdrant vector store                    │
       │              (./qdrant_storage/)                    │
       └─────────────────────────────────────────────────────┘
                                                       │
                                                       ▼
                                              ┌────────────────────┐
                                              │ ChatOllama llama3  │
                                              │  → final answer    │
                                              └────────────────────┘
```

The pipeline is built using LangChain's
[runnables](https://python.langchain.com/docs/expression_language/) /
LCEL composition — retrieval → context formatting → prompt → LLM.

---

## Project structure

```
bfsi-policy-assistant/
├── app.py             # Streamlit app + RAG pipeline (single file)
├── requirements.txt   # Python dependencies
├── .gitignore         # Ignores .venv, qdrant_storage/, uploaded docs
├── LICENSE            # MIT
└── README.md
```

At runtime, the app creates:

```
├── qdrant_storage/    # Persisted vector store (git-ignored)
```

---

## Notes & limitations

- **First indexing pass is slow.** Embedding a large policy document on
  CPU via Ollama can take several minutes. Subsequent launches reuse the
  Qdrant collection, so it's instant after that.
- **GPU strongly recommended** for `llama3` answers. CPU works, just
  expect each answer to take 10–60+ seconds depending on context length.
- **No multi-user isolation.** Everyone hitting the same Streamlit
  instance shares the same Qdrant collection. Fine for personal use; not
  suitable for a multi-tenant deployment as-is.
- **Document files are git-ignored by default.** If you want to ship a
  sample document with the repo, use `git add -f sample.pdf`.

---

## License

[MIT](./LICENSE)
