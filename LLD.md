# BFSI Policy Assistant: LLD

## How the pieces fit

```
upload --> loaders --> heading-aware chunker --> char-level chunker
                                                      |
                                                      v
                              local Ollama embeddings (llama3)
                                                      |
                                                      v
                                  Chroma at ./chroma_db_storage
                                                      ^
                                                      |
question --> query expansion --> sim + MMR retrieval -+
                                                      |
                                                      v
                         dedupe + rerank (term overlap + heading bonus)
                                                      |
                                                      v
                          top 15 chunks + question --> OpenRouter (Mistral 7B)
                                                      |
                                                      v
                                                  Streamlit
```

Everything lives in one file (`app.py`). I considered splitting into modules. For around 500 lines it's not worth the navigation cost.

## The classes that matter

`OptimizedChromaDBManager` owns the Chroma vectorstore's lifecycle: create, close, wipe. The "optimized" prefix is leftover from an earlier round; what it actually does is batch the persist calls and gc between batches, because building the store in one shot blows up memory at 5000+ chunks. It also retries `shutil.rmtree` up to 5 times with a 1-second backoff, because Windows holds file locks for a few seconds after Chroma releases them.

`HierarchicalChunker` does two passes. `split_by_headings` walks the document line by line and groups lines under the most recent detected heading. The heading regex is:

```python
[
    r'^\d+\.\s+',                              # "1. Foo"
    r'^\d+\.\d+\.\s+',                          # "2.3. Bar"
    r'^[A-Z][A-Z\s]{3,}$',                      # "ALL CAPS HEADING"
    r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:$',       # "Section Title:"
]
```

Anything matching, and shorter than 100 chars, is treated as a heading. The 100-char cap is a hack to prevent body lines that happen to start with a number from being treated as section heads.

`smart_chunking` then runs RecursiveCharacterTextSplitter on any section longer than 900 chars (1.5 times CHUNK_SIZE). Shorter sections are left whole, which keeps short clauses (a one-paragraph "Definitions" section, say) coherent.

`OptimizedRetriever` is the one to read carefully. The interesting bit is `_smart_query_expansion`: from a question like "what is the cancellation window for term insurance", it generates:

1. The full question.
2. The top-2 keyword pair: "cancellation window".
3. The top-3 keyword triple: "cancellation window term".
4. Individual long keywords: "cancellation", "insurance".

Capped at 5 variants, of which the first 3 are actually run through retrieval. The cap is empirical; beyond 3 the marginal recall stops paying for the latency.

After retrieval, candidates are deduped by `md5(chunk.page_content)`, then re-ranked. The rerank score is the count of question terms (filtered against a 19-word stopword set, minimum length 3) appearing in the chunk, plus 3 if the chunk has a parent heading. I considered cross-encoder reranking; for v1 the term-overlap + heading bonus does well enough and adds zero dependencies.

## Constants worth knowing

```
DB_DIR              = "./chroma_db_storage"
COLLECTION_NAME     = "bfsi_documents_full"
MODEL_NAME          = "mistralai/mistral-7b-instruct-v0.1"
EMBEDDING_MODEL     = "llama3"
MAX_CANDIDATE_CHUNKS = 60   # candidates fetched before rerank
FINAL_CONTEXT_CHUNKS = 15   # what actually goes into the prompt
CHUNK_SIZE          = 600
CHUNK_OVERLAP       = 150
BATCH_SIZE          = 50    # embed and add this many at a time
```

`CHUNK_SIZE = 600` is the one tuned specifically for BFSI. The default in tutorials is 1000. Policy text packs more meaning per character than prose, so 600 keeps each chunk to roughly one self-contained clause.

`FINAL_CONTEXT_CHUNKS = 15` times 600 chars works out to about 9000 chars, roughly 2200 tokens. Mistral 7B Instruct's context is 8K. That leaves around 5800 tokens for the prompt template and the answer. Comfortable.

## What gets carried in metadata

```python
{
    "source_name": "policy_xyz.pdf",
    "file_type": "PDF",
    "doc_id": "<uuid4>",
    "processed_date": "2026-06-13 14:22:08",
    "page": 3,
    "heading": "4.2 Cancellation Window",
}
```

`source_name` and `page` end up in citations. `heading` ends up in the rerank score and in citations. `doc_id` and `processed_date` are for debugging multi-file uploads; nothing user-facing uses them.

## The two paths through the system

### Loading a corpus

1. User drops up to 5 files into the uploader, clicks Load.
2. For each file: write to OS temp, dispatch to the right loader, stamp metadata.
3. `split_by_headings` walks every doc, attaching `heading`.
4. `smart_chunking` re-splits oversize sections.
5. `OptimizedChromaDBManager.create_vectorstore`:
   - Close any existing store, gc.
   - `rmtree(DB_DIR)` with retries.
   - Create the store from the first 50 chunks.
   - For remaining chunks, `add_documents` in 50-batches, updating the progress bar.
6. Wrap in `OptimizedRetriever`, stash in `st.session_state`.

### Asking a question

1. User types into chat input.
2. `get_comprehensive_context`:
   - Expand into up to 5 variants, run the first 3.
   - For each variant: similarity (k=30) + MMR (k=20).
   - Dedupe by md5.
   - Score (term overlap + heading bonus), sort, take 15.
   - Format each chunk as `Excerpt N | Document: ... | Page: ... | Section: ...`.
3. Build the prompt: `HUMAN_PROMPT.format(context=..., question=...)`.
4. Send to OpenRouter via `ChatOpenAI`.
5. Render response in chat, append to `messages`.

## Where it breaks

| What happens | What the app does |
|---|---|
| `OPENROUTER_API_KEY` missing | `st.error` plus `st.stop` on first generation attempt |
| Ollama not running | First embedding call raises; the Streamlit error panel shows the exception |
| PDF has no text layer | Chunker yields zero chunks; the load succeeds with a misleading "0 chunks" message (known UX bug) |
| Chroma directory locked (Windows) | `rmtree` retries 5 times with 1 second backoff |
| OpenRouter 429 or 5xx | `ChatOpenAI` retries 3 times; after that, `st.error` |
| Empty retrieval | Context string is empty, the LLM is **not** called, UI shows "no relevant sections" |

## Privacy boundary

This is the one slide I'd show to a security reviewer.

- Embeddings: local. Document text never leaves the host at index time.
- Storage: local. Chunk text lives in `./chroma_db_storage/` on the user's disk. Protect that directory like any other source of sensitive data.
- Retrieval: local.
- Generation: third party. The retrieved 15 chunks plus the question go to OpenRouter. Nothing else.

If a customer can't accept the generation step's third-party hop, swap the generator for local Ollama and you've got a fully on-host pipeline. The cost is generation quality.

## Things worth doing next

In rough priority order:

1. **Don't wipe on Load.** Detect file-hash collisions, skip re-embedding files that are already indexed. This is the single biggest UX improvement.
2. **PII redaction pre-LLM.** A regex pass over the context (PAN, Aadhaar, IFSC, account numbers). Cheap and reduces the third-party risk meaningfully.
3. **Generator interface.** Right now `get_llm()` is one function pointing at OpenRouter. Refactor into a `Generator` protocol so swapping to local-only is a one-line change.
4. **Multi-collection.** Promote `COLLECTION_NAME` to per-corpus, add a corpus picker to the sidebar.
