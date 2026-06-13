# BFSI Policy Assistant: PRD

## Why this exists

I work around BFSI policy docs all the time and the failure mode is depressingly consistent: someone needs an answer from a 200-page PDF, they Ctrl+F a phrase that doesn't quite match the document's wording, they give up, and either ask a senior colleague or guess. The "guess" path is what generates the regulatory pain. Ctrl+F is the wrong tool for natural-language questions, and asking the same five people the same five questions doesn't scale.

A general-purpose chatbot doesn't fix this either, because the cost of a confident wrong answer is much higher than the cost of "I don't know". Anything we build has to refuse cleanly when the answer isn't in the documents.

That's the wedge: a RAG assistant whose answer is always grounded in retrieved passages, never in the model's general knowledge, and whose citations point to a specific document, page, and section.

## Who it's for

Three users I had in mind while building this:

1. **The frontline support agent.** They get a customer question, they have 30 seconds, they need to cite the right clause. Today they fumble through PDFs while the customer waits.
2. **The compliance analyst.** They're cross-checking whether an internal procedure matches a stated policy. The bottleneck is finding the policy text, not interpreting it.
3. **The new hire in week one.** Onboarding decks always lag the real policies. They need a way to ask "what does our travel-claim window actually look like" and get the document, not a summary slide.

## What "done" looks like for v1

The hard requirements:

- Every answer cites at least one document, with page and section heading.
- The model never makes up content. If retrieval returns nothing useful, the response is some flavor of "I couldn't find that in the documents you uploaded."
- A single user can run the whole system on their laptop. No server, no Docker, no auth wall. Ollama + Streamlit + an OpenRouter free-tier key.
- Document text stays on the host at embedding time. The retrieval step never sends documents to a third party.

What I am explicitly not building in v1:

- Multi-tenant isolation. One user, one corpus.
- Real-time corpus updates. Loading is a manual step.
- Scanned PDFs. If your PDF needs OCR, run OCR first.
- Conversation memory across sessions. Each session is fresh.
- Multi-language support. English only.

A note I want to be honest about: I'd originally written that "the index persists across restarts so you don't re-embed". The code persists the Chroma directory, but the Load button always wipes and rebuilds. So the persistence is really crash recovery, not incremental indexing. Worth fixing in v2; today it's a footnote, not a feature.

## How a user gets through the day

The flow is uneventful, which is the point.

1. They pull `ollama llama3`, grab an OpenRouter key, put it in `.env`, source it, run `streamlit run app.py`.
2. They drag up to five files into the upload widget. The hint says 40-50 pages each, which keeps the chunk count below a few thousand and the latency reasonable. They click Load.
3. While it embeds (around 30 seconds for 250 pages on my laptop), they sip coffee.
4. They ask a question. The answer arrives with a "Sources Used" expander showing which documents were cited and a "Retrieval Report" expander showing which chunks fed the prompt. The expanders matter: they let the user catch the times the answer is grounded in the wrong source.

Clearing state is a separate button. I considered making it automatic on file change and decided no, because reuploading the same files is a common debugging move.

## The hard requirements (numbered for traceability)

Ingestion:

- Up to five files per load, in PDF, DOCX, or TXT.
- Page numbers come from PDF and DOCX loaders. TXT has none.
- Every chunk carries `source_name`, `file_type`, `doc_id` (uuid), and `processed_date`.

Chunking:

- Heading-aware split first. The regex catches numeric prefixes ("1. ", "2.3. "), short ALL-CAPS lines, and Title-Case lines ending in a colon.
- Within each section, RecursiveCharacterTextSplitter at 600 chars with 150 overlap. Smaller than the usual 1000 because policy text is denser than prose.
- Every chunk keeps its parent heading.

Indexing:

- Embedding via local Ollama `llama3`.
- Persistent to `./chroma_db_storage/`, collection `bfsi_documents_full`.
- Batches of 50 to bound memory.
- Loading a new corpus wipes the existing collection first.

Retrieval:

- Up to 5 query variants, first 3 actually run.
- Each variant does similarity_search (k=30) and MMR (k=20, fetch_k=30, lambda_mult=0.6).
- Dedupe candidates by md5 of chunk text.
- Score is `term_overlap + 3 if chunk has a heading`.
- Top 15 to the prompt.

Generation:

- Mistral 7B Instruct v0.1 via OpenRouter, temperature 0.2, 120 second timeout, 3 retries.
- Prompt template forbids out-of-context answers and requires inline citations.

## What I'd watch in production

Numbers I'd want a dashboard for, if this were past v1:

- Retrieval hit rate: does the top-15 contain the gold passage? Target 90% on a held-out set.
- Citation correctness: every cited section actually exists. Should be 100%; below that and the prompt has drifted.
- P95 end-to-end latency. On my machine, 250 pages, about 12 seconds. Anything above 20 and something's wrong with Ollama or OpenRouter.

## What could go wrong

The four I lose sleep over:

1. **OpenRouter goes down or rate-limits me.** The fix is to swap the generator behind a thin interface and fall back to a local Ollama model. Worth doing pre-launch if anyone but me uses this.
2. **Confidential clauses fly to a third party at answer time.** This is true by design. The retrieved excerpts go to OpenRouter. Future work: a redaction pass over the context before the LLM call (PAN, Aadhaar, IFSC, account numbers).
3. **The heading regex misfires.** If a body line looks like a heading, the chunker treats it as one and the section bonus gets misapplied. The 100-char length cap mitigates this, not perfectly.
4. **Someone uploads a scanned PDF.** PyPDFLoader returns empty pages, the chunker yields zero chunks, the user sees "0 chunks" and rightly wonders what happened. Today the warning is too quiet; it should be louder.

## Open

- Should the Retrieval Report be on by default? Today it's behind an expander; I lean toward visible-by-default for the first 10 questions, then collapsible.
- Top-k of 15 is a code constant. Should it be a slider? My instinct: no, it adds a knob users don't know how to set.
- Compare-two-documents mode. People have asked. I'd want a real use case before building.
