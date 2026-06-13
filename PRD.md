# BFSI Policy Assistant: Product Requirements Document

> Version 1.0 - authored alongside the v1 implementation in `app.py`.

## 1. Problem

Banks, insurers, and NBFCs own large bodies of internal policy text: product T&Cs, claim-handling rules, KYC procedures, audit policies. Frontline staff and compliance analysts need fast, citable answers from these documents. Today they choose between three poor options:

1. Search PDFs manually with Ctrl+F, which fails when the question is phrased differently from the document.
2. Ask a senior colleague, which is slow and creates a knowledge bottleneck.
3. Use a general-purpose chatbot, which hallucinates or quotes the wrong policy.

The cost of the wrong answer is real: misquoted clauses lead to regulatory issues, mis-sold products, and customer disputes. A retrieval-augmented assistant that grounds every answer in a specific clause solves the right problem.

## 2. Target users

| User | Need | Current pain |
|---|---|---|
| Frontline support agent | Answer a customer query in under a minute, citing the exact clause | Slow manual search; fear of misquoting |
| Compliance analyst | Verify that internal procedure X matches policy Y | Cross-document lookup is manual and error-prone |
| New hire (week one) | Build a mental model of the policy landscape | Onboarding decks lag the real policy text |

## 3. Goals (must-have)

1. Every answer is grounded in retrieved passages from the documents the user actually uploaded. No external knowledge claims.
2. Every answer is traceable: document name, page number, section heading.
3. The vector index persists across restarts so re-launching the app does not trigger re-embedding.
4. The host machine does the embedding work. Document text is not sent to a third party at index time.
5. The free tier of OpenRouter is sufficient to operate the system for a single user.

## 4. Non-goals (out of scope for v1)

1. Multi-tenant isolation. The app runs as one user with one corpus.
2. Real-time corpus updates (webhooks, file watchers).
3. Fine-tuning on the user's corpus.
4. Conversation memory across sessions. Each session starts clean.
5. Multi-language documents. English only.
6. Tabular or image-heavy documents (charts, scanned PDFs without a text layer).

## 5. User journeys

### 5.1 First-time setup

1. User installs Ollama and pulls the `llama3` model.
2. User registers for an OpenRouter account (free tier) and copies the API key.
3. User copies `.env.example` to `.env`, fills in `OPENROUTER_API_KEY`, sources it, and runs `streamlit run app.py`.

### 5.2 Loading a corpus

1. User opens the app and drags up to 5 files (PDF, DOCX, or TXT, around 40 to 50 pages each) into the upload widget.
2. User clicks "Load Documents".
3. The app shows a per-file load progress bar, then chunks and embeds them in batches of 50.
4. The sidebar surfaces total chunks and pages per file. The app is now ready for questions.

### 5.3 Asking a question

1. User types a question, for example "what is the cancellation window for policy X?".
2. The app expands the query into variations, runs retrieval, and picks the top 15 chunks.
3. The app calls Mistral 7B via OpenRouter with the retrieved chunks and the question.
4. The chat panel shows the answer; an expander lists the source documents used; a retrieval report shows which excerpts were chosen.

### 5.4 Clearing state

1. User clicks "Clear All".
2. The Chroma directory is removed; session state is reset; the app returns to the welcome screen.

## 6. Functional requirements

### Ingestion

- **F1.** Accept up to 5 files per load, of types PDF, DOCX, TXT.
- **F2.** Loaders attach a 1-based page number for PDF and DOCX. Plain text has no page concept.
- **F3.** Per-file metadata stamps every chunk with `source_name`, `file_type`, `doc_id` (uuid4), and `processed_date`.

### Chunking

- **F4.** Heading-aware split first. Detect headings by regex: numeric prefix (`1. `, `2.3. `), ALL-CAPS lines under 100 chars, and Title-Case lines ending with a colon.
- **F5.** Within each detected section, further split with `RecursiveCharacterTextSplitter` at a target size of 600 characters and 150 character overlap, respecting natural breaks (paragraph, sentence, clause).
- **F6.** Each chunk carries its parent heading in metadata so citations can render "Section: 4.2 Cancellation Window".

### Indexing

- **F7.** Embed each chunk with the local Ollama `llama3` model.
- **F8.** Persist vectors to a Chroma collection at `./chroma_db_storage/` with collection name `bfsi_documents_full`.
- **F9.** Embed and add in batches of 50 chunks to bound memory.
- **F10.** Loading a new corpus wipes the existing collection first (no append, no version mismatch).

### Retrieval

- **F11.** Expand the user query into up to 5 variants (full question, top-2 keyword pair, top-3 keyword triple, individual long keywords).
- **F12.** For the first 3 variants, run both similarity search (k=30) and MMR search (k=20, fetch_k=30, lambda_mult=0.6).
- **F13.** Deduplicate candidates by MD5 of the chunk content.
- **F14.** Score each unique candidate as `count_of_question_terms_in_chunk + 3 if chunk_has_heading else 0`. Sort descending.
- **F15.** Take the top 15 chunks as final context.

### Generation

- **F16.** Prompt template instructs the model to answer only from the provided excerpts, cite source and page, and fall back to a "not found" message when the answer is not present.
- **F17.** Model is `mistralai/mistral-7b-instruct-v0.1` via OpenRouter, temperature 0.2, timeout 120 seconds, up to 3 retries.

### UI

- **F18.** Sidebar surfaces the file uploader, Load button, Clear button, and per-file stats once loaded.
- **F19.** Main panel surfaces chat history and chat input.
- **F20.** Each assistant message exposes a "Sources Used" expander and a "Retrieval Report" expander.

## 7. Non-functional requirements

- **N1. Privacy boundary.** Document text is embedded locally; only the retrieved top-15 excerpts plus the question are sent to OpenRouter. Users must understand this boundary; the README documents it.
- **N2. Cost.** The free tier of OpenRouter is sufficient for personal use; prompt size is bounded by the chunk size and top-k.
- **N3. Resilience.** OpenRouter calls retry up to 3 times. After that the error surfaces to the user, not swallowed.
- **N4. Latency target.** For a 250-page corpus, an answer should arrive in under 15 seconds on a laptop with 16 GB RAM and Ollama running locally.
- **N5. Operational simplicity.** No database server, no Docker, no auth layer. One Python process plus Ollama.

## 8. Success metrics

| Metric | Target |
|---|---|
| Retrieval contains the correct passage in the top 15 | 90% of held-out questions |
| Citations match real document text (no fabricated quotes) | 100% |
| User-reported "correct answer" on a 50-question pilot | over 85% |
| P95 end-to-end latency on a 250-page corpus | under 15 seconds |

## 9. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| OpenRouter outage breaks generation | Medium | Wrap the generator behind a small interface and swap to a local Ollama generator |
| Confidential clauses go to a third party at answer time | Always true by design | Documented in README; future work to add a PII redaction pass before the LLM call |
| User uploads scanned PDFs without a text layer | Medium | Detect empty extraction and warn the user; OCR is out of scope for v1 |
| Free-tier quota exhausted | Low for a single user | Surface 429 verbatim; user can switch keys |
| Heading regex misclassifies normal lines as headings | Medium | 100-character cap; fall back to the character splitter for oversize sections |

## 10. Open questions

1. Should the retrieval report be hidden by default, or always visible to build user trust?
2. Do users want a "compare two documents" mode?
3. Is 15 chunks the right top-k? Today it is a code constant, not a UI control.
