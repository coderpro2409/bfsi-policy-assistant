# BFSI Policy Assistant: Low-Level Design

> Version 1.0 - describes the implementation in `app.py` as it stands today.

## 1. Architecture overview

```
+-------------+      +-----------------+      +---------------+
| Streamlit   | ---> | Document        | ---> | Hierarchical  |
| File upload |      | loaders         |      | chunker       |
+-------------+      +-----------------+      +-------+-------+
                                                      |
                                                      v
                     +-----------------+      +-----------------+
                     | Ollama          | <--- | OptimizedChroma |
                     | (llama3 embed)  |      | DBManager       |
                     +-----------------+      +-----------------+
                                                      |
                                                      v
+-------------+      +-----------------+      +-----------------+
| Streamlit   | ---> | OptimizedRetri- | ---> | Chroma          |
| Chat input  |      | ever            | <--- | persisted store |
+-------------+      +-------+---------+      +-----------------+
                             |
                             v
                     +-----------------+
                     | OpenRouter      |
                     | (Mistral 7B)    |
                     +-----------------+
                             |
                             v
                     +-----------------+
                     | Streamlit       |
                     | Chat output     |
                     +-----------------+
```

## 2. Modules in `app.py`

| Module / Class | Responsibility |
|---|---|
| `get_llm(temperature)` | Build a `ChatOpenAI` client pointed at OpenRouter; halt with a clear UI error if `OPENROUTER_API_KEY` is missing |
| `OptimizedChromaDBManager` | Own the Chroma vectorstore lifecycle: close, create with batched persist, wipe |
| `HierarchicalChunker.split_by_headings` | Detect headings by regex, group lines into sections, attach `heading` to each section's metadata |
| `HierarchicalChunker.smart_chunking` | For sections longer than 1.5x `CHUNK_SIZE`, apply `RecursiveCharacterTextSplitter` |
| `OptimizedRetriever._smart_query_expansion` | Build up to 5 query variants from the question's keywords |
| `OptimizedRetriever.get_comprehensive_context` | Run similarity + MMR over the first 3 variants, dedupe by MD5, re-rank by term overlap and heading bonus, return the top-15 chunks plus a sources list |
| `load_document_with_pages` | Dispatch to `PyPDFLoader`, `Docx2txtLoader`, or `TextLoader`; attach a page number per fragment |
| `process_documents` | Save uploads to temp files, load, attach uniform metadata, run the chunker, return chunks and per-file stats |

## 3. Configuration constants

| Constant | Value | Reason |
|---|---|---|
| `DB_DIR` | `./chroma_db_storage` | Persistence path for the Chroma collection |
| `COLLECTION_NAME` | `bfsi_documents_full` | One fixed collection at a time |
| `MODEL_NAME` | `mistralai/mistral-7b-instruct-v0.1` | Free-tier model on OpenRouter |
| `EMBEDDING_MODEL` | `llama3` | Local Ollama embedding model |
| `MAX_CANDIDATE_CHUNKS` | 60 | Upper bound on candidates fetched before re-ranking |
| `FINAL_CONTEXT_CHUNKS` | 15 | Top-k after re-ranking; balances recall and prompt size |
| `CHUNK_SIZE` | 600 chars | Smaller than typical (1000) because BFSI policy text is dense |
| `CHUNK_OVERLAP` | 150 chars | Preserves sentence boundaries across cuts |
| `BATCH_SIZE` | 50 | Per-batch embed and add, to bound memory |

## 4. Data shapes

### 4.1 Chunk metadata

```python
{
    "source_name": "policy_xyz.pdf",
    "file_type": "PDF" | "DOCX" | "TXT",
    "doc_id": "<uuid4>",
    "processed_date": "YYYY-MM-DD HH:MM:SS",
    "page": int | None,
    "heading": str | None,
}
```

### 4.2 Retrieved context block (string sent to the LLM)

```
**Excerpt 1**
Document: policy_xyz.pdf | Page 3 | Section: 4.2 Cancellation Window
<chunk text>

---

**Excerpt 2**
Document: policy_xyz.pdf | Page 7 | Section: 5.1 Refund Eligibility
<chunk text>
```

### 4.3 Session state keys

| Key | Type | Lifetime |
|---|---|---|
| `retriever` | `OptimizedRetriever \| None` | Per session |
| `documents_loaded` | bool | Per session |
| `db_manager` | `OptimizedChromaDBManager` | Per session |
| `messages` | list of `{role, content, sources}` | Per session |
| `total_chunks` | int | Per session |
| `file_stats` | list of `{name, pages, type}` | Per session |

## 5. Sequence: load documents

1. UI: user uploads up to 5 files and clicks "Load Documents".
2. `process_documents`: for each file, write to OS temp, dispatch to the right loader, attach metadata.
3. `HierarchicalChunker.split_by_headings`: regex-detect section heads, group lines, attach `heading`.
4. `HierarchicalChunker.smart_chunking`: re-split any section larger than `CHUNK_SIZE * 1.5`.
5. `OptimizedChromaDBManager.create_vectorstore`:
   1. Close any current store; gc.
   2. `shutil.rmtree(DB_DIR)` with up to 5 retries (handles Windows file locks).
   3. Create the store from the first batch of 50 chunks.
   4. For remaining chunks, call `add_documents` in batches of 50, with a progress bar.
6. Wrap the store in `OptimizedRetriever`; persist in session state.

## 6. Sequence: answer a question

1. UI: user types a question and submits.
2. `OptimizedRetriever.get_comprehensive_context(question)`:
   1. Build up to 5 query variants via `_smart_query_expansion`.
   2. For the first 3 variants:
      - `similarity_search(q, k=30)`
      - `max_marginal_relevance_search(q, k=20, fetch_k=30, lambda_mult=0.6)`
   3. Dedupe candidates by `md5(chunk.page_content)`.
   4. Score: count of question terms (length above 2, not in stopword list) appearing in the chunk text, plus 3 if the chunk has a heading.
   5. Sort by score; take top 15.
   6. Format each chunk as a labeled excerpt with `Document | Page | Section`.
3. Build the prompt from `HUMAN_PROMPT.format(context=..., question=...)`.
4. Send the prompt to OpenRouter via `ChatOpenAI`.
5. Render the response in the chat panel; append to `messages`.

## 7. Failure modes

| Scenario | Behavior |
|---|---|
| `OPENROUTER_API_KEY` missing | `get_llm` calls `st.error` then `st.stop`; the user sees the env var hint |
| Ollama not running | First `OllamaEmbeddings` call raises; the error surfaces in the Streamlit error panel |
| PDF has no extractable text | `PyPDFLoader` returns empty pages; chunker yields zero chunks; the app shows a warning |
| Chroma DB directory locked (Windows) | Retry `rmtree` up to 5 times with 1-second backoff |
| OpenRouter 429 / 5xx | `ChatOpenAI` retries up to 3 times; if it still fails, the exception is rendered via `st.error` |
| Empty retrieval | Context is empty; the UI shows "No relevant sections found"; the LLM is **not** called |

## 8. Security and privacy

- The only secret is `OPENROUTER_API_KEY`, read from the environment.
- Uploaded files are written to OS temp via `tempfile.NamedTemporaryFile` and deleted after loading.
- Chunk text lives on disk in `./chroma_db_storage/`; the user is responsible for protecting that directory.
- At answer time, only the top-15 retrieved excerpts plus the question are sent to OpenRouter; the rest of the corpus stays on the host.
- No telemetry; no analytics; no logs of question content beyond the in-memory chat history.

## 9. Performance notes

- Embedding cost is one-time per load; persistence covers crash recovery, not incremental indexing (loading a new corpus wipes the store).
- 15 chunks of 600 chars is roughly 9000 chars or 2000 to 2500 tokens of context. Mistral 7B Instruct handles 8K tokens, leaving comfortable headroom for the prompt template and the answer.
- The stopword list in scoring is hand-curated. Swapping in a tokenizer-based filter would improve precision on short, common-word questions.

## 10. Extension points

- **Generator swap.** Replace `get_llm()` to call a local Ollama generation model. The rest of the pipeline is unaffected.
- **PII pre-filter.** Add a redaction pass over `context` before the LLM call to strip PAN, Aadhaar, IFSC, account numbers.
- **Persistent chat sessions.** Replace `st.session_state.messages` with a sqlite cache to survive page reloads.
- **Multi-collection.** Promote `COLLECTION_NAME` to per-corpus and add a corpus picker in the sidebar.

## 11. Operational runbook

| Task | Command |
|---|---|
| Pull the embedding model | `ollama pull llama3` |
| Start Ollama | `ollama serve` (or run it as a system service) |
| Configure the API key | `cp .env.example .env`, edit `.env`, then `set -a; source .env; set +a` |
| Run the app | `streamlit run app.py` |
| Reset the index | Click "Clear All" in the sidebar, or `rm -rf ./chroma_db_storage` |
