import streamlit as st
import os
import tempfile
import shutil
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings
from datetime import datetime
from typing import List, Tuple
import hashlib
import time
import re
import uuid
from collections import defaultdict
import gc

# OpenRouter API key is read from the OPENROUTER_API_KEY environment variable.
# Copy .env.example to .env, fill in your key, then load it before running:
#   set -a; source .env; set +a
#   streamlit run app.py
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="BFSI Policy Assistant - Enterprise Scale",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0A4D8C 0%, #1E88E5 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .info-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F57C00;
        margin: 1rem 0;
    }
    .stat-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
DB_DIR = "./chroma_db_storage"
COLLECTION_NAME = "bfsi_documents_full"
MODEL_NAME = "mistralai/mistral-7b-instruct-v0.1"   # Free tier
EMBEDDING_MODEL = "llama3"                       # Local embeddings

MAX_CANDIDATE_CHUNKS = 60
FINAL_CONTEXT_CHUNKS = 15
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150
BATCH_SIZE = 50

# --- OPENROUTER LLM SETUP ---
def get_llm(temperature=0.2):
    if not OPENROUTER_API_KEY:
        st.error(
            "OPENROUTER_API_KEY is not set. "
            "Copy .env.example to .env, add your key, then run: "
            "`set -a; source .env; set +a; streamlit run app.py`"
        )
        st.stop()
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model=MODEL_NAME,
        temperature=temperature,
        timeout=120,
        max_retries=3,
        default_headers={
            "HTTP-Referer": "https://bfsi-assistant.local",
            "X-Title": "BFSI Document Assistant"
        }
    )

# --- CHROMADB MANAGER (optimized) ---
class OptimizedChromaDBManager:
    def __init__(self):
        self.vectorstore = None
        
    def close_connection(self):
        if self.vectorstore:
            try:
                self.vectorstore = None
                gc.collect()
            except:
                pass
    
    def create_vectorstore(self, documents, embeddings):
        try:
            self.close_connection()
            if os.path.exists(DB_DIR):
                for attempt in range(5):
                    try:
                        shutil.rmtree(DB_DIR)
                        time.sleep(0.5)
                        break
                    except PermissionError:
                        time.sleep(1)
            
            total_docs = len(documents)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            first_batch = documents[:BATCH_SIZE]
            vectorstore = Chroma.from_documents(
                documents=first_batch,
                embedding=embeddings,
                persist_directory=DB_DIR,
                collection_name=COLLECTION_NAME
            )
            
            for i in range(BATCH_SIZE, total_docs, BATCH_SIZE):
                batch = documents[i:i+BATCH_SIZE]
                vectorstore.add_documents(batch)
                progress = min((i + BATCH_SIZE) / total_docs, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Processing documents: {int(progress * 100)}%")
                gc.collect()
            
            progress_bar.empty()
            status_text.empty()
            return vectorstore
        except Exception as e:
            st.error(f"Error creating vectorstore: {e}")
            return None

# --- HIERARCHICAL CHUNKING ---
class HierarchicalChunker:
    @staticmethod
    def split_by_headings(docs):
        chunked_docs = []
        for doc in docs:
            content = doc.page_content
            metadata = doc.metadata.copy()
            lines = content.split('\n')
            current_heading = "Introduction"
            current_section = []
            
            for line in lines:
                heading_patterns = [
                    r'^\d+\.\s+',
                    r'^\d+\.\d+\.\s+',
                    r'^[A-Z][A-Z\s]{3,}$',
                    r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:$'
                ]
                is_heading = any(re.match(pattern, line) for pattern in heading_patterns)
                
                if is_heading and len(line.strip()) < 100:
                    if current_section:
                        section_text = '\n'.join(current_section)
                        new_meta = metadata.copy()
                        new_meta['heading'] = current_heading
                        chunked_docs.append(type('Document', (), {
                            'page_content': section_text,
                            'metadata': new_meta
                        })())
                    current_heading = line.strip()
                    current_section = [line]
                else:
                    current_section.append(line)
            
            if current_section:
                section_text = '\n'.join(current_section)
                new_meta = metadata.copy()
                new_meta['heading'] = current_heading
                chunked_docs.append(type('Document', (), {
                    'page_content': section_text,
                    'metadata': new_meta
                })())
        return chunked_docs if chunked_docs else docs
    
    @staticmethod
    def smart_chunking(section_docs):
        final_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
        )
        for doc in section_docs:
            if len(doc.page_content) <= CHUNK_SIZE * 1.5:
                final_chunks.append(doc)
            else:
                chunks = text_splitter.split_documents([doc])
                final_chunks.extend(chunks)
        return final_chunks

# --- OPTIMIZED RETRIEVER ---
class OptimizedRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.stop_words = {'how', 'what', 'when', 'where', 'why', 'will', 'does', 'do', 'is', 'are',
                          'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'to', 'for', 'of',
                          'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'if'}
    
    def _smart_query_expansion(self, question):
        words = question.lower().split()
        key_terms = [w for w in words if w not in self.stop_words and len(w) > 3]
        queries = [question]
        if len(key_terms) >= 2:
            queries.append(f"{key_terms[0]} {key_terms[1]}")
        if len(key_terms) >= 3:
            queries.append(f"{key_terms[0]} {key_terms[1]} {key_terms[2]}")
        for term in key_terms[:3]:
            if len(term) > 5:
                queries.append(term)
        return list(set(queries))[:5]
    
    def _extract_heading(self, content):
        lines = content.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if re.match(r'^\d+\.\s+', line) or (line.isupper() and len(line) < 80) or line.endswith(':'):
                return line
        return None
    
    def get_comprehensive_context(self, question):
        queries = self._smart_query_expansion(question)
        all_candidates = []
        sources = set()
        
        for q in queries[:3]:
            try:
                sim_docs = self.vectorstore.similarity_search(q, k=MAX_CANDIDATE_CHUNKS//2)
                all_candidates.extend(sim_docs)
                mmr_docs = self.vectorstore.max_marginal_relevance_search(
                    q, k=MAX_CANDIDATE_CHUNKS//3, fetch_k=MAX_CANDIDATE_CHUNKS//2, lambda_mult=0.6
                )
                all_candidates.extend(mmr_docs)
            except:
                continue
        
        unique = []
        seen = set()
        for doc in all_candidates:
            h = hashlib.md5(doc.page_content.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(doc)
                sources.add(doc.metadata.get('source_name', 'Unknown'))
        
        question_terms = set(question.lower().split())
        question_terms = {t for t in question_terms if t not in self.stop_words and len(t) > 2}
        
        scored = []
        for doc in unique:
            content_lower = doc.page_content.lower()
            term_score = sum(1 for term in question_terms if term in content_lower)
            heading = doc.metadata.get('heading', '')
            heading_bonus = 3 if heading else 0
            total = term_score + heading_bonus
            scored.append((total, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        final_docs = [doc for score, doc in scored[:FINAL_CONTEXT_CHUNKS]]
        
        formatted = []
        details = []
        for i, doc in enumerate(final_docs):
            source = doc.metadata.get('source_name', 'Unknown')
            page = doc.metadata.get('page', None)
            heading = doc.metadata.get('heading', None) or self._extract_heading(doc.page_content)
            content = doc.page_content.strip()
            
            page_str = f"Page {page}" if page and page != 'Not specified' else ""
            heading_str = f"Section: {heading}" if heading else ""
            ref_parts = [f"Document: {source}"]
            if page_str:
                ref_parts.append(page_str)
            if heading_str:
                ref_parts.append(heading_str)
            reference = " | ".join(ref_parts)
            
            formatted.append(f"**Excerpt {i+1}**\n{reference}\n{content}\n")
            details.append({
                "source": source,
                "page": page,
                "heading": heading,
                "excerpt": content[:300] + "..."
            })
        
        context = "\n\n---\n\n".join(formatted)
        sources_list = list(set(d['source'] for d in details))
        
        with st.expander("🔍 Retrieval Report (Optimized)"):
            st.write(f"**Documents covered:** {len(sources_list)}")
            st.write(f"**Selected excerpts:** {len(final_docs)}")
            for d in details[:5]:
                st.caption(f"• {d['source']} | Page {d['page']} | {d['heading']}")
        
        return context, sources_list, details

# --- DOCUMENT LOADING ---
def load_document_with_pages(file_path, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata['page'] = i + 1
        return docs
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata['page'] = i + 1
        return docs
    else:
        loader = TextLoader(file_path)
        docs = loader.load()
        return docs

def process_documents(uploaded_files):
    all_docs = []
    file_stats = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(uploaded_files):
        ext = file.name.split(".")[-1].lower()
        status_text.text(f"Loading: {file.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file.getvalue())
            path = tmp.name
        
        docs = load_document_with_pages(path, ext)
        for d in docs:
            d.metadata["source_name"] = file.name
            d.metadata["file_type"] = ext.upper()
            d.metadata["doc_id"] = str(uuid.uuid4())
            d.metadata["processed_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        all_docs.extend(docs)
        file_stats.append({
            "name": file.name,
            "pages": len(docs),
            "type": ext.upper()
        })
        os.remove(path)
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    status_text.text("Chunking documents...")
    hierarchical_chunker = HierarchicalChunker()
    section_docs = hierarchical_chunker.split_by_headings(all_docs)
    final_chunks = hierarchical_chunker.smart_chunking(section_docs)
    
    status_text.text(f"Created {len(final_chunks)} chunks")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    return final_chunks, file_stats, len(final_chunks)

# --- HUMANIZED PROMPT ---
HUMAN_PROMPT = """
You are a friendly, knowledgeable BFSI assistant. Answer the customer's question **only** using the provided document excerpts.

**Instructions:**
- Write in a warm, helpful tone – as if you're talking to a client.
- **Always cite your sources** using the exact document name and page number (e.g., "According to the Policy Document, page 3...").
- If the same information appears in multiple places, combine it into one complete answer.
- Be thorough but concise. Focus on what the customer actually asked.
- If the answer is not in the excerpts, say: "I couldn't find that in the documents you uploaded. Could you rephrase or check if the information exists?"
- Do NOT include technical markers like [Excerpt 1] or brackets. Just give a natural answer.

**Document excerpts (with sources):**
{context}

**Customer question:** {question}

**Your helpful answer:**
"""

# --- MAIN UI ---
st.markdown("""
<div class="main-header">
    <h1>🏦 BFSI Document Assistant</h1>
    <p>Enterprise Scale | Powered by Qwen via OpenRouter | Handles 250+ Pages</p>
</div>
""", unsafe_allow_html=True)

# Session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    st.session_state.documents_loaded = False
    st.session_state.db_manager = OptimizedChromaDBManager()
    st.session_state.messages = []
    st.session_state.total_chunks = 0
    st.session_state.file_stats = []

# Sidebar
with st.sidebar:
    st.markdown("## 📚 Document Management")
    files = st.file_uploader("Upload Documents (PDF, DOCX, TXT)", accept_multiple_files=True,
                             type=['pdf', 'docx', 'txt'], help="Up to 5 files, 40-50 pages each")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Load Documents", use_container_width=True):
            if files and len(files) <= 5:
                with st.spinner("Processing large documents..."):
                    chunks, file_stats, chunk_count = process_documents(files)
                    if chunks:
                        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
                        vs = st.session_state.db_manager.create_vectorstore(chunks, embeddings)
                        if vs:
                            st.session_state.retriever = OptimizedRetriever(vs)
                            st.session_state.documents_loaded = True
                            st.session_state.total_chunks = chunk_count
                            st.session_state.file_stats = file_stats
                            st.session_state.messages = []
                            st.success(f"✅ Loaded {len(files)} document(s) → {chunk_count} chunks")
                            for stat in file_stats:
                                st.info(f"📄 {stat['name']}: {stat['pages']} pages")
                            st.rerun()
                    else:
                        st.error("Processing failed. Please try again.")
            elif len(files) > 5:
                st.warning("Maximum 5 files at a time.")
            else:
                st.warning("Please select files first")
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            if os.path.exists(DB_DIR):
                for attempt in range(5):
                    try:
                        shutil.rmtree(DB_DIR)
                        break
                    except PermissionError:
                        time.sleep(1)
            st.session_state.retriever = None
            st.session_state.documents_loaded = False
            st.session_state.messages = []
            st.session_state.total_chunks = 0
            st.session_state.file_stats = []
            st.success("Cleared!")
            st.rerun()
    
    st.markdown("---")
    if st.session_state.documents_loaded:
        st.success("✅ Documents ready")
        st.metric("Total Chunks", st.session_state.total_chunks)
        for stat in st.session_state.file_stats:
            st.caption(f"📄 {stat['name']}: {stat['pages']} pages")
    else:
        st.info("📭 No documents loaded")

# Main chat area
if not st.session_state.documents_loaded:
    st.markdown("""
    <div class="info-box">
        <h3>👋 Enterprise Document Assistant</h3>
        <p>Optimized for <strong>5 files × 50 pages (250+ pages total)</strong> with:</p>
        <ul>
            <li>🔍 Hierarchical chunking (preserves section structure)</li>
            <li>⚡ Batch processing for memory efficiency</li>
            <li>🤖 Qwen via OpenRouter (privacy-focused, no training on your data)</li>
            <li>📄 Page numbers and section headings preserved</li>
            <li>💬 Humanized responses with citations</li>
        </ul>
        <p><strong>Upload your documents to begin!</strong></p>
    </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources Used"):
                    for src in message["sources"]:
                        st.markdown(f"- {src}")

    if user_input := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching through documents..."):
                try:
                    context, sources, details = st.session_state.retriever.get_comprehensive_context(user_input)
                    if not context:
                        st.warning("No relevant sections found. Please try rephrasing.")
                    else:
                        llm = get_llm(temperature=0.2)
                        prompt = ChatPromptTemplate.from_template(HUMAN_PROMPT)
                        chain = prompt | llm | StrOutputParser()
                        response = chain.invoke({"context": context, "question": user_input})
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources
                        })
                except Exception as e:
                    st.error(f"Error: {e}")

st.markdown("---")
st.caption("⚡ Enterprise optimized | Qwen via OpenRouter (private) | Handles 250+ pages efficiently")