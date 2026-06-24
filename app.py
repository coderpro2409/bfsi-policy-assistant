# app.py - BFSI Multi-Tenant Document Assistant (Complete Production Version)
import streamlit as st
import os
import shutil
import json
import hashlib
import time
import re
import uuid
from datetime import datetime
from pathlib import Path
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import threading
import webbrowser

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==================== CONFIGURATION ====================
# Hosted LLM via OpenRouter (no local Ollama needed for cloud deploy).
# Set OPENROUTER_API_KEY in your host's secrets (Streamlit Cloud / HF Spaces).
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")
# Embeddings run in-process on CPU (free, no API key, works on Streamlit Cloud).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

BASE_DIR = Path("./bfsi_multi_tenant")
BASE_DIR.mkdir(exist_ok=True)
COMPANIES_DIR = BASE_DIR / "companies"
COMPANIES_DIR.mkdir(exist_ok=True)
VECTOR_DB_DIR = BASE_DIR / "vector_dbs"
VECTOR_DB_DIR.mkdir(exist_ok=True)
COMPANIES_JSON = BASE_DIR / "companies.json"
CLIENTS_JSON = BASE_DIR / "clients.json"
CHAT_HISTORY_JSON = BASE_DIR / "chat_history.json"

MAX_CANDIDATE_CHUNKS = 60
FINAL_CONTEXT_CHUNKS = 15
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ==================== JSON STORAGE ====================
def load_companies():
    if COMPANIES_JSON.exists():
        with open(COMPANIES_JSON, 'r') as f:
            return json.load(f)
    return {}

def save_companies(companies):
    with open(COMPANIES_JSON, 'w') as f:
        json.dump(companies, f, indent=2)

def load_clients():
    if CLIENTS_JSON.exists():
        with open(CLIENTS_JSON, 'r') as f:
            return json.load(f)
    return {}

def save_clients(clients):
    with open(CLIENTS_JSON, 'w') as f:
        json.dump(clients, f, indent=2)

def load_chat_history():
    if CHAT_HISTORY_JSON.exists():
        with open(CHAT_HISTORY_JSON, 'r') as f:
            return json.load(f)
    return {}

def save_chat_history(history):
    with open(CHAT_HISTORY_JSON, 'w') as f:
        json.dump(history, f, indent=2)

# ==================== MIGRATION FOR OLD CHAT HISTORY ====================
def migrate_old_chat_history():
    """Convert old chat history format to new format with messages array."""
    if not CHAT_HISTORY_JSON.exists():
        return
    try:
        with open(CHAT_HISTORY_JSON, 'r') as f:
            ch = json.load(f)
    except:
        return
    modified = False
    for company_name, sessions in ch.items():
        if isinstance(sessions, list):
            new_sessions = []
            for sess in sessions:
                if isinstance(sess, dict) and 'question' in sess and 'answer' in sess:
                    # Old format: session has question/answer directly
                    new_sess = {
                        "session_id": sess.get('session_id', str(uuid.uuid4())),
                        "client_email": "anonymous",
                        "client_name": "Guest",
                        "start_time": sess.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        "messages": [{
                            "question": sess['question'],
                            "answer": sess['answer'],
                            "timestamp": sess.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            "sources": sess.get('sources', '')
                        }]
                    }
                    new_sessions.append(new_sess)
                    modified = True
                else:
                    # Already new format, ensure messages is a list
                    if not isinstance(sess.get('messages'), list):
                        sess['messages'] = []
                    new_sessions.append(sess)
            if modified:
                ch[company_name] = new_sessions
    if modified:
        with open(CHAT_HISTORY_JSON, 'w') as f:
            json.dump(ch, f, indent=2)
        print("✅ Chat history migrated to new format.")

# ==================== LLM / EMBEDDING HELPERS ====================
def get_llm(temperature=0.1):
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your host's secrets "
            "(Streamlit Cloud: App settings -> Secrets). Get a free key at "
            "https://openrouter.ai/keys"
        )
    return ChatOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=OPENROUTER_API_KEY,
        model=LLM_MODEL,
        temperature=temperature,
        max_tokens=2048,
    )

# Cache the embedding model so the (~90MB) weights load only once per session.
_embeddings_cache = None
def get_embeddings():
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings_cache

# ==================== SEMANTIC CHUNKER ====================
class SemanticChunker:
    @staticmethod
    def split_documents(docs, chunk_size=1000, overlap=200):
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=separators,
            length_function=len,
        )
        all_chunks = []
        for doc in docs:
            chunks = text_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata['original_heading'] = doc.metadata.get('heading', 'Unknown')
            all_chunks.extend(chunks)
        return all_chunks

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

def process_documents_for_company(company_name, uploaded_files):
    company_dir = COMPANIES_DIR / company_name
    company_dir.mkdir(exist_ok=True)
    all_docs = []
    file_stats = []
    for file_data in uploaded_files:
        filename = file_data['name']
        file_content = file_data['content']
        ext = filename.split(".")[-1].lower()
        file_path = company_dir / filename
        with open(file_path, 'wb') as f:
            f.write(file_content)
        docs = load_document_with_pages(str(file_path), ext)
        for d in docs:
            d.metadata["source_name"] = filename
            d.metadata["file_type"] = ext.upper()
            d.metadata["doc_id"] = str(uuid.uuid4())
            d.metadata["processed_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        all_docs.extend(docs)
        file_stats.append({"name": filename, "pages": len(docs), "type": ext.upper()})

    chunker = SemanticChunker()
    final_chunks = chunker.split_documents(all_docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    for idx, chunk in enumerate(final_chunks):
        chunk.id = f"{company_name}_{uuid.uuid4().hex}_{idx}"

    vector_db_path = VECTOR_DB_DIR / company_name
    collection_name = re.sub(r'[^a-zA-Z0-9._-]', '', company_name.replace(' ', '_')) + "_documents"
    embeddings = get_embeddings()

    if vector_db_path.exists() and any(vector_db_path.iterdir()):
        try:
            vectorstore = Chroma(persist_directory=str(vector_db_path), embedding_function=embeddings, collection_name=collection_name)
            vectorstore.add_documents(final_chunks)
        except Exception as e:
            print(f"Error adding to existing DB, recreating: {e}")
            shutil.rmtree(vector_db_path, ignore_errors=True)
            vectorstore = Chroma.from_documents(final_chunks, embeddings, persist_directory=str(vector_db_path), collection_name=collection_name)
    else:
        vectorstore = Chroma.from_documents(final_chunks, embeddings, persist_directory=str(vector_db_path), collection_name=collection_name)
    return vectorstore, file_stats, len(final_chunks)

# ==================== LOAD EXISTING VECTOR STORES ====================
def load_existing_retrievers():
    companies = load_companies()
    retrievers_loaded = 0
    for company_name, company_data in companies.items():
        if company_data.get('documents_loaded'):
            vector_db_path = VECTOR_DB_DIR / company_name
            if vector_db_path.exists():
                try:
                    collection_name = re.sub(r'[^a-zA-Z0-9._-]', '', company_name.replace(' ', '_')) + "_documents"
                    embeddings = get_embeddings()
                    vectorstore = Chroma(persist_directory=str(vector_db_path), embedding_function=embeddings, collection_name=collection_name)
                    company_retrievers[company_name] = OptimizedRetriever(vectorstore)
                    retrievers_loaded += 1
                except Exception as e:
                    print(f"Failed to load vector store for {company_name}: {e}")
    print(f"Loaded {retrievers_loaded} existing vector stores")

# ==================== OPTIMIZED RETRIEVER ====================
class OptimizedRetriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.stop_words = {'how','what','when','where','why','will','does','do','is','are','was','were','the','a','an','and','or','but','to','for','of','with','by','from','up','about','into','through','during','if'}

    def _smart_query_expansion(self, question):
        words = question.lower().split()
        key_terms = [w for w in words if w not in self.stop_words and len(w) > 3]
        queries = [question]
        if len(key_terms) >= 2: queries.append(f"{key_terms[0]} {key_terms[1]}")
        if len(key_terms) >= 3: queries.append(f"{key_terms[0]} {key_terms[1]} {key_terms[2]}")
        for term in key_terms[:3]:
            if len(term) > 5: queries.append(term)
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
                sim = self.vectorstore.similarity_search(q, k=MAX_CANDIDATE_CHUNKS//2)
                mmr = self.vectorstore.max_marginal_relevance_search(q, k=MAX_CANDIDATE_CHUNKS//3, fetch_k=MAX_CANDIDATE_CHUNKS//2, lambda_mult=0.6)
                all_candidates.extend(sim)
                all_candidates.extend(mmr)
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

        question_terms = {t for t in question.lower().split() if t not in self.stop_words and len(t)>2}
        scored = []
        for doc in unique:
            content_lower = doc.page_content.lower()
            term_score = sum(1 for t in question_terms if t in content_lower)
            heading_bonus = 3 if doc.metadata.get('heading') else 0
            scored.append((term_score + heading_bonus, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        final_docs = [doc for _, doc in scored[:FINAL_CONTEXT_CHUNKS]]

        formatted = []
        details = []
        for i, doc in enumerate(final_docs):
            source = doc.metadata.get('source_name', 'Unknown')
            page = doc.metadata.get('page')
            heading = doc.metadata.get('heading') or self._extract_heading(doc.page_content)
            content = doc.page_content.strip()
            ref = f"**{source}**" + (f", page {page}" if page else "") + (f", section: {heading}" if heading else "")
            formatted.append(f"**Excerpt {i+1}**\n{ref}\n{content}\n")
            details.append({"source": source, "page": page, "heading": heading, "excerpt": content[:300]})

        context = "\n\n---\n\n".join(formatted)
        sources_list = list(set(d['source'] for d in details))
        return context, sources_list, details

# ==================== STRICT PROMPT (No Ellipsis, No Inference) ====================
HUMAN_PROMPT = """
You are a precise BFSI assistant. Answer the question **only** using the provided document excerpts.

**ABSOLUTE RULES – VIOLATION WILL CONFUSE USERS:**
1. **NEVER use "..." (ellipsis) anywhere in your answer.** Quote full sentences or complete clauses.
2. **NEVER say "it can be inferred that"** – if the document does not directly state something, say: "The documents do not explicitly state this. They do say: [full quote of relevant clause]."
3. **Always quote the exact text** from the excerpts. Use quotation marks and cite the document name and page number.
4. **If the answer is not in the excerpts**, say exactly: "I cannot find that information in the provided documents."
5. **Do not summarize** – let the document speak for itself.

**Excerpts:**
{context}

**Question:** {question}

**Answer (full quotes, no ellipsis, no inferences):**
"""

# ==================== FLASK APP ====================
flask_app = Flask(__name__)
flask_app.secret_key = os.getenv('FLASK_SECRET_KEY', 'change-this-secret-key')
flask_app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
flask_app.config['SESSION_COOKIE_HTTPONLY'] = True
# Same-origin in the cloud (Flask serves both the HTML and the /api routes),
# so CORS origins are only needed for local split-port development.
CORS(flask_app, supports_credentials=True, origins=['http://localhost:5000', 'http://localhost:8501'])

company_retrievers = {}

# ---------- COMPANY PORTAL HTML (Full) ----------
COMPANY_PORTAL_HTML = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BFSI Company Portal</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 { color: #667eea; margin-bottom: 10px; }
        .card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: 600; color: #333; }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 600;
        }
        button:hover { transform: translateY(-2px); }
        .status { padding: 10px; border-radius: 5px; margin-top: 10px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .info { background: #d1ecf1; color: #0c5460; }
        .nav-tabs {
            display: flex;
            margin-bottom: 20px;
            border-bottom: 2px solid #ddd;
        }
        .nav-tab {
            padding: 10px 20px;
            cursor: pointer;
            background: none;
            border: none;
            font-size: 16px;
            color: #666;
        }
        .nav-tab.active { color: white; border-bottom: 2px solid #667eea; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        /* Chat history grouping styles */
        .client-group {
            margin-bottom: 25px;
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
        }
        .client-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .client-name {
            font-size: 18px;
            font-weight: 700;
            color: #333;
            background: #f0f0f0;
            padding: 5px 12px;
            border-radius: 20px;
        }
        .client-badge {
            background: #667eea;
            color: white;
            padding: 4px 10px;
            border-radius: 15px;
            font-size: 12px;
        }
        .filter-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 20px;
        }
        .filter-btn {
            background: #e0e0e0;
            color: #333;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            border: none;
            font-size: 14px;
        }
        .filter-btn.active {
            background: #667eea;
            color: white;
        }
        .chat-history-item {
            border-left: 3px solid #667eea;
            padding-left: 15px;
            margin-bottom: 20px;
        }
        .message-block {
            background: #f9f9f9;
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .no-history {
            color: #666;
            font-style: italic;
            padding: 15px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏦 BFSI Company Portal</h1>
            <p>Manage your documents and get your Q&A bot link</p>
        </div>
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('login')">Login</button>
            <button class="nav-tab" onclick="showTab('register')">Register</button>
        </div>
        <div id="login" class="tab-content active">
            <div class="card">
                <h2>Company Login</h2>
                <form id="loginForm">
                    <div class="form-group">
                        <label>Email</label>
                        <input type="email" id="loginEmail" required>
                    </div>
                    <div class="form-group">
                        <label>Password</label>
                        <input type="password" id="loginPassword" required>
                    </div>
                    <button type="submit">Login</button>
                </form>
                <div id="loginStatus"></div>
            </div>
        </div>
        <div id="register" class="tab-content">
            <div class="card">
                <h2>Company Registration</h2>
                <form id="registerForm">
                    <div class="form-group">
                        <label>Company Name</label>
                        <input type="text" id="regCompanyName" required>
                    </div>
                    <div class="form-group">
                        <label>Email</label>
                        <input type="email" id="regEmail" required>
                    </div>
                    <div class="form-group">
                        <label>Password</label>
                        <input type="password" id="regPassword" required>
                    </div>
                    <div class="form-group">
                        <label>Enable Chat History</label>
                        <select id="regChatHistory">
                            <option value="true">Yes</option>
                            <option value="false">No</option>
                        </select>
                    </div>
                    <button type="submit">Register</button>
                </form>
                <div id="registerStatus"></div>
            </div>
        </div>
        <div id="dashboard" style="display:none;">
            <div class="card">
                <h2>Welcome, <span id="companyName"></span>!</h2>
                <button onclick="logout()" style="background:#dc3545; margin-top:10px;">Logout</button>
            </div>
            <div class="card">
                <h3>📄 Upload Documents</h3>
                <form id="uploadForm">
                    <div class="form-group">
                        <label>Select Documents (PDF, DOCX, TXT - Max 5 files)</label>
                        <input type="file" id="documents" multiple accept=".pdf,.docx,.txt">
                    </div>
                    <button type="submit">Upload Documents</button>
                </form>
                <div id="uploadStatus"></div>
            </div>
            <div class="card">
                <h3>🚀 Process Documents</h3>
                <p>After uploading, click below to process documents and create the Q&A bot.</p>
                <button onclick="processDocuments()">Load & Process Documents</button>
                <div id="processStatus"></div>
                <div id="companyLink" style="margin-top:20px; display:none;">
                    <h4>Your Q&A Bot Link:</h4>
                    <input type="text" id="botLink" readonly style="background:#f0f0f0; width:100%;">
                    <button onclick="copyLink()" style="margin-top:10px;">Copy Link</button>
                </div>
            </div>
            <div id="chatHistoryCard" class="card" style="display:none;">
                <h3>💬 Chat History (CRM View)</h3>
                <div id="chatHistoryFilters" class="filter-buttons"></div>
                <div id="chatHistoryList"></div>
            </div>
        </div>
    </div>
    <script>
        let rawHistoryData = [];
        let currentFilter = 'all';
        
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            document.querySelectorAll('.nav-tab').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }
        async function fetchAPI(url, options = {}) {
            options.credentials = 'include';
            if (!options.headers) options.headers = {};
            if (!(options.body instanceof FormData)) options.headers['Content-Type'] = 'application/json';
            return fetch(url, options);
        }
        document.getElementById('registerForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const data = {
                company_name: document.getElementById('regCompanyName').value,
                email: document.getElementById('regEmail').value,
                password: document.getElementById('regPassword').value,
                chat_history_enabled: document.getElementById('regChatHistory').value === 'true'
            };
            const response = await fetchAPI('/api/company/register', { method: 'POST', body: JSON.stringify(data) });
            const result = await response.json();
            const statusDiv = document.getElementById('registerStatus');
            if (response.ok) {
                statusDiv.innerHTML = '<div class="status success">✅ Registration successful! Please login.</div>';
                setTimeout(() => showTab('login'), 2000);
            } else {
                statusDiv.innerHTML = `<div class="status error">❌ ${result.error}</div>`;
            }
        });
        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const data = { email: document.getElementById('loginEmail').value, password: document.getElementById('loginPassword').value };
            const response = await fetchAPI('/api/company/login', { method: 'POST', body: JSON.stringify(data) });
            const result = await response.json();
            const statusDiv = document.getElementById('loginStatus');
            if (response.ok) {
                statusDiv.innerHTML = '<div class="status success">✅ Login successful!</div>';
                document.getElementById('companyName').innerText = result.company.company_name;
                document.getElementById('login').style.display = 'none';
                document.getElementById('register').style.display = 'none';
                document.querySelector('.nav-tabs').style.display = 'none';
                document.getElementById('dashboard').style.display = 'block';
                const isEnabled = result.company.chat_history_enabled === true || result.company.chat_history_enabled === 'true';
                if (isEnabled) loadChatHistory();
            } else {
                statusDiv.innerHTML = `<div class="status error">❌ ${result.error}</div>`;
            }
        });
        async function loadChatHistory() {
            try {
                const response = await fetch('/api/company/chat_history', { credentials: 'include' });
                const result = await response.json();
                if (response.ok && result.history) {
                    rawHistoryData = result.history;
                    renderChatHistory();
                } else {
                    console.error("Failed to load chat history");
                    document.getElementById('chatHistoryCard').style.display = 'none';
                }
            } catch(err) {
                console.error(err);
                document.getElementById('chatHistoryCard').style.display = 'none';
            }
        }
        
        function renderChatHistory() {
            const container = document.getElementById('chatHistoryList');
            const filterContainer = document.getElementById('chatHistoryFilters');
            if (!container) return;
            
            // Group sessions by client (email)
            const clientMap = new Map(); // key: client_email, value: { name, email, sessions, isAnonymous }
            for (const session of rawHistoryData) {
                const email = session.client_email || 'anonymous';
                const name = session.client_name || (email === 'anonymous' ? 'Anonymous User' : email);
                const isAnonymous = (email === 'anonymous');
                if (!clientMap.has(email)) {
                    clientMap.set(email, { name, email, isAnonymous, sessions: [] });
                }
                clientMap.get(email).sessions.push(session);
            }
            
            // Build filter buttons
            filterContainer.innerHTML = '';
            const allBtn = document.createElement('button');
            allBtn.className = 'filter-btn' + (currentFilter === 'all' ? ' active' : '');
            allBtn.textContent = 'All Clients';
            allBtn.onclick = () => { currentFilter = 'all'; renderChatHistory(); };
            filterContainer.appendChild(allBtn);
            
            // Add buttons for registered clients (non-anonymous)
            for (let [email, data] of clientMap.entries()) {
                if (!data.isAnonymous) {
                    const btn = document.createElement('button');
                    btn.className = 'filter-btn' + (currentFilter === email ? ' active' : '');
                    btn.textContent = data.name;
                    btn.onclick = () => { currentFilter = email; renderChatHistory(); };
                    filterContainer.appendChild(btn);
                }
            }
            // Add anonymous button if exists
            if (clientMap.has('anonymous')) {
                const btn = document.createElement('button');
                btn.className = 'filter-btn' + (currentFilter === 'anonymous' ? ' active' : '');
                btn.textContent = 'Anonymous Users';
                btn.onclick = () => { currentFilter = 'anonymous'; renderChatHistory(); };
                filterContainer.appendChild(btn);
            }
            
            // Filter sessions
            let filteredSessions = [];
            if (currentFilter === 'all') {
                for (let sessions of clientMap.values()) {
                    filteredSessions.push(...sessions.sessions);
                }
            } else {
                const clientData = clientMap.get(currentFilter);
                if (clientData) filteredSessions = clientData.sessions;
            }
            
            if (filteredSessions.length === 0) {
                container.innerHTML = '<div class="no-history">No chat history for this filter.</div>';
                document.getElementById('chatHistoryCard').style.display = 'block';
                return;
            }
            
            // Display sessions grouped by client (when showing all) or directly when filtered
            container.innerHTML = '';
            if (currentFilter === 'all') {
                // Group by client again for display
                for (let [email, data] of clientMap.entries()) {
                    if (data.sessions.length === 0) continue;
                    const groupDiv = document.createElement('div');
                    groupDiv.className = 'client-group';
                    groupDiv.innerHTML = `
                        <div class="client-header">
                            <span class="client-name">${escapeHtml(data.name)}</span>
                            <span class="client-badge">${data.isAnonymous ? 'Anonymous' : 'Registered Client'}</span>
                            <span>${data.sessions.length} session(s)</span>
                        </div>
                    `;
                    for (const session of data.sessions) {
                        const sessionDiv = document.createElement('div');
                        sessionDiv.className = 'chat-history-item';
                        sessionDiv.innerHTML = `<strong>Session started: ${escapeHtml(session.start_time)}</strong>`;
                        const messages = Array.isArray(session.messages) ? session.messages : [];
                        for (const msg of messages) {
                            const msgDiv = document.createElement('div');
                            msgDiv.className = 'message-block';
                            msgDiv.innerHTML = `
                                <div><strong>❓ Question:</strong> ${escapeHtml(msg.question)}</div>
                                <div><strong>💬 Answer:</strong> ${escapeHtml(msg.answer)}</div>
                                <div><strong>📚 Sources:</strong> ${escapeHtml(msg.sources)}</div>
                                <div style="font-size:12px; color:#666;">🕒 ${escapeHtml(msg.timestamp)}</div>
                            `;
                            sessionDiv.appendChild(msgDiv);
                        }
                        groupDiv.appendChild(sessionDiv);
                    }
                    container.appendChild(groupDiv);
                }
            } else {
                // Single client view
                for (const session of filteredSessions) {
                    const sessionDiv = document.createElement('div');
                    sessionDiv.className = 'chat-history-item';
                    sessionDiv.innerHTML = `<strong>Session started: ${escapeHtml(session.start_time)}</strong>`;
                    const messages = Array.isArray(session.messages) ? session.messages : [];
                    for (const msg of messages) {
                        const msgDiv = document.createElement('div');
                        msgDiv.className = 'message-block';
                        msgDiv.innerHTML = `
                            <div><strong>❓ Question:</strong> ${escapeHtml(msg.question)}</div>
                            <div><strong>💬 Answer:</strong> ${escapeHtml(msg.answer)}</div>
                            <div><strong>📚 Sources:</strong> ${escapeHtml(msg.sources)}</div>
                            <div style="font-size:12px; color:#666;">🕒 ${escapeHtml(msg.timestamp)}</div>
                        `;
                        sessionDiv.appendChild(msgDiv);
                    }
                    container.appendChild(sessionDiv);
                }
            }
            document.getElementById('chatHistoryCard').style.display = 'block';
        }
        
        function escapeHtml(text) { if (!text) return ''; const div = document.createElement('div'); div.textContent = text; return div.innerHTML; }
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const files = document.getElementById('documents').files;
            if (files.length === 0 || files.length > 5) { alert('Please select 1-5 files'); return; }
            const formData = new FormData();
            for (let i=0; i<files.length; i++) formData.append('documents', files[i]);
            const statusDiv = document.getElementById('uploadStatus');
            statusDiv.innerHTML = '<div class="status info">📤 Uploading...</div>';
            const response = await fetch('/api/company/upload', { method: 'POST', credentials: 'include', body: formData });
            const result = await response.json();
            if (response.ok) statusDiv.innerHTML = '<div class="status success">✅ Uploaded!</div>';
            else statusDiv.innerHTML = `<div class="status error">❌ ${result.error}</div>`;
        });
        async function processDocuments() {
            const statusDiv = document.getElementById('processStatus');
            statusDiv.innerHTML = '<div class="status info">🔄 Processing... <div class="loading"></div></div>';
            const response = await fetch('/api/company/process', { method: 'POST', credentials: 'include' });
            const result = await response.json();
            if (response.ok) {
                statusDiv.innerHTML = '<div class="status success">✅ Processed!</div>';
                document.getElementById('companyLink').style.display = 'block';
                document.getElementById('botLink').value = `${window.location.origin}/company/${result.company_url}`;
            } else {
                statusDiv.innerHTML = `<div class="status error">❌ ${result.error}</div>`;
            }
        }
        function copyLink() { const link = document.getElementById('botLink'); link.select(); document.execCommand('copy'); alert('Link copied!'); }
        async function logout() { await fetch('/api/company/logout', { method: 'POST', credentials: 'include' }); location.reload(); }
    </script>
</body>
</html>
'''
# ---------- CLIENT PORTAL HTML (Full) ----------
CLIENT_PORTAL_HTML = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>BFSI Client Portal</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 { color: #667eea; }
        .nav-tabs { display: flex; justify-content: center; margin-bottom: 20px; }
        .nav-tab { padding: 10px 30px; cursor: pointer; background: white; border: none; margin: 0 5px; border-radius: 5px; font-weight: 600; }
        .nav-tab.active { background: #667eea; color: white; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            max-width: 400px;
            margin: 0 auto;
        }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: 600; color: #333; }
        input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px;
            border-radius: 5px;
            cursor: pointer;
            width: 100%;
            font-size: 16px;
            font-weight: 600;
        }
        .status { padding: 10px; border-radius: 5px; margin-top: 10px; text-align: center; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .info { background: #d1ecf1; color: #0c5460; }
        .companies-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .company-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.2s;
        }
        .company-card:hover { transform: translateY(-5px); }
        .company-name { font-size: 20px; font-weight: 600; color: #667eea; margin-bottom: 15px; }
        .company-actions { display: flex; gap: 10px; justify-content: center; }
        .view-docs-btn { background: #28a745; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer; }
        .chat-container {
            background: white;
            border-radius: 10px;
            height: 600px;
            display: flex;
            flex-direction: column;
        }
        .chat-header {
            padding: 15px;
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 10px 10px 0 0;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .message { margin-bottom: 15px; display: flex; }
        .message.user { justify-content: flex-end; }
        .message-content {
            max-width: 70%;
            padding: 10px 15px;
            border-radius: 10px;
        }
        .message.user .message-content { background: #667eea; color: white; }
        .message.assistant .message-content { background: #f0f0f0; color: #333; }
        .chat-input {
            padding: 20px;
            border-top: 1px solid #ddd;
            display: flex;
            gap: 10px;
        }
        .chat-input input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .chat-input button {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            width: auto;
        }
        .back-button {
            background: #6c757d;
            margin-bottom: 20px;
            width: auto;
        }
        .sources {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
            border-top: 1px solid #eee;
            padding-top: 5px;
        }
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 10px 15px;
        }
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background-color: #999;
            border-radius: 50%;
            animation: bounce 1.4s infinite;
        }
        @keyframes bounce { 0%,60%,100% { transform: translateY(0); } 30% { transform: translateY(-10px); } }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.7);
        }
        .modal-content {
            background-color: white;
            margin: 5% auto;
            padding: 20px;
            width: 80%;
            height: 80%;
            border-radius: 10px;
            overflow-y: auto;
            position: relative;
        }
        .close-modal {
            position: absolute;
            right: 20px;
            top: 10px;
            font-size: 28px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏦 BFSI Client Portal</h1>
            <p>Ask questions to companies about their documents</p>
        </div>
        <div class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('clientLogin')">Login</button>
            <button class="nav-tab" onclick="showTab('clientRegister')">Register</button>
        </div>
        <div id="clientLogin" class="tab-content active">
            <div class="card">
                <h2>Client Login</h2>
                <form id="clientLoginForm">
                    <div class="form-group"><label>Email</label><input type="email" id="clientLoginEmail" required></div>
                    <div class="form-group"><label>Password</label><input type="password" id="clientLoginPassword" required></div>
                    <button type="submit">Login</button>
                </form>
                <div id="clientLoginStatus"></div>
            </div>
        </div>
        <div id="clientRegister" class="tab-content">
            <div class="card">
                <h2>Client Registration</h2>
                <form id="clientRegisterForm">
                    <div class="form-group"><label>Full Name</label><input type="text" id="clientName" required></div>
                    <div class="form-group"><label>Email</label><input type="email" id="clientRegEmail" required></div>
                    <div class="form-group"><label>Password</label><input type="password" id="clientRegPassword" required></div>
                    <button type="submit">Register</button>
                </form>
                <div id="clientRegisterStatus"></div>
            </div>
        </div>
        <div id="clientDashboard" style="display:none;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h2>Welcome, <span id="clientNameDisplay"></span>!</h2>
                <button onclick="clientLogout()" style="background:#dc3545; width:auto;">Logout</button>
            </div>
            <div id="companiesView">
                <div class="companies-grid" id="companiesList"></div>
            </div>
            <div id="chatView" style="display:none;">
                <button class="back-button" onclick="showCompanies()">← Back to Companies</button>
                <div class="chat-container">
                    <div class="chat-header">
                        <strong id="chatCompanyName"></strong>
                        <button class="view-docs-btn" onclick="viewDocuments()">📄 View Documents</button>
                    </div>
                    <div class="chat-messages" id="chatMessages"></div>
                    <div class="chat-input">
                        <input type="text" id="questionInput" placeholder="Ask a question..." onkeypress="if(event.key==='Enter') askQuestion()">
                        <button onclick="askQuestion()">Send</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div id="docModal" class="modal">
        <div class="modal-content">
            <span class="close-modal" onclick="closeDocModal()">&times;</span>
            <h3 id="modalTitle">Document Viewer</h3>
            <div id="modalContent"></div>
        </div>
    </div>
    <script>
        let currentCompany = null;
        let sessionId = Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.getElementById(tabName).classList.add('active');
            document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
        }
        async function fetchAPI(url, options = {}) {
            options.credentials = 'include';
            if (!options.headers) options.headers = {};
            if (!(options.body instanceof FormData)) options.headers['Content-Type'] = 'application/json';
            return fetch(url, options);
        }
        document.getElementById('clientRegisterForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const data = {
                name: document.getElementById('clientName').value,
                email: document.getElementById('clientRegEmail').value,
                password: document.getElementById('clientRegPassword').value
            };
            const response = await fetchAPI('/api/client/register', { method: 'POST', body: JSON.stringify(data) });
            const result = await response.json();
            const statusDiv = document.getElementById('clientRegisterStatus');
            if (response.ok) {
                statusDiv.innerHTML = '<div class="status success">✅ Registration successful! Please login.</div>';
                setTimeout(() => showTab('clientLogin'), 2000);
            } else {
                statusDiv.innerHTML = `<div class="status error">❌ ${result.error}</div>`;
            }
        });
        document.getElementById('clientLoginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const data = {
                email: document.getElementById('clientLoginEmail').value,
                password: document.getElementById('clientLoginPassword').value
            };
            const response = await fetchAPI('/api/client/login', { method: 'POST', body: JSON.stringify(data) });
            const result = await response.json();
            const statusDiv = document.getElementById('clientLoginStatus');
            if (response.ok) {
                statusDiv.innerHTML = '<div class="status success">✅ Login successful!</div>';
                document.getElementById('clientNameDisplay').innerText = result.client.name;
                document.getElementById('clientLogin').style.display = 'none';
                document.getElementById('clientRegister').style.display = 'none';
                document.querySelector('.nav-tabs').style.display = 'none';
                document.getElementById('clientDashboard').style.display = 'block';
                loadCompanies();
            } else {
                statusDiv.innerHTML = `<div class="status error">❌ ${result.error}</div>`;
            }
        });
        async function loadCompanies() {
            const response = await fetch('/api/companies');
            const companies = await response.json();
            const container = document.getElementById('companiesList');
            container.innerHTML = '';
            for (const company of companies) {
                const card = document.createElement('div');
                card.className = 'company-card';
                card.innerHTML = `
                    <div class="company-name">🏢 ${company.company_name}</div>
                    <div class="company-actions">
                        <button class="view-docs-btn" onclick="event.stopPropagation(); viewDocumentsFromCard('${company.company_name}', '${company.company_url}')">📄 View Documents</button>
                    </div>
                `;
                card.onclick = () => openChat(company);
                container.appendChild(card);
            }
        }
        async function viewDocumentsFromCard(name, url) { currentCompany = { company_name: name, company_url: url }; await viewDocuments(); }
        async function viewDocuments() {
            if (!currentCompany) return;
            const response = await fetch(`/api/company/${currentCompany.company_url}/documents`);
            const docs = await response.json();
            let html = '<h4>Available Documents (Read Only)</h4><ul>';
            for (const doc of docs) html += `<li><strong>${doc.filename}</strong> - <button onclick="viewDocumentContent('${doc.filename}')">View</button></li>`;
            html += '</ul>';
            document.getElementById('modalTitle').innerHTML = `${currentCompany.company_name} - Documents`;
            document.getElementById('modalContent').innerHTML = html;
            document.getElementById('docModal').style.display = 'block';
        }
        async function viewDocumentContent(filename) {
            const response = await fetch(`/api/company/${currentCompany.company_url}/document/${encodeURIComponent(filename)}`);
            const content = await response.text();
            document.getElementById('modalContent').innerHTML = `<pre style="white-space: pre-wrap;">${escapeHtml(content)}</pre>`;
        }
        function escapeHtml(text) { const div = document.createElement('div'); div.textContent = text; return div.innerHTML; }
        function closeDocModal() { document.getElementById('docModal').style.display = 'none'; }
        async function openChat(company) {
            currentCompany = company;
            document.getElementById('companiesView').style.display = 'none';
            document.getElementById('chatView').style.display = 'block';
            document.getElementById('chatCompanyName').innerText = company.company_name;
            document.getElementById('chatMessages').innerHTML = '';
            addMessage('assistant', `Hello! I'm the assistant for ${company.company_name}. Ask me anything about their documents.`);
        }
        function showCompanies() { currentCompany = null; document.getElementById('companiesView').style.display = 'block'; document.getElementById('chatView').style.display = 'none'; }
        async function askQuestion() {
            const input = document.getElementById('questionInput');
            const question = input.value.trim();
            if (!question) return;
            input.value = '';
            addMessage('user', question);
            const typingId = addTypingIndicator();
            try {
                const response = await fetch('/api/chat/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ company_name: currentCompany.company_name, session_id: sessionId, question: question })
                });
                const result = await response.json();
                removeTypingIndicator(typingId);
                if (response.ok) {
                    let answer = result.answer;
                    if (result.sources && result.sources.length) answer += `<div class="sources">📚 Sources: ${result.sources.join(', ')}</div>`;
                    addMessage('assistant', answer);
                } else { addMessage('assistant', `Error: ${result.error || 'Unable to get answer'}`); }
            } catch(err) { removeTypingIndicator(typingId); addMessage('assistant', 'Error: Could not connect to server. Make sure Ollama is running.'); }
        }
        function addTypingIndicator() {
            const container = document.getElementById('chatMessages');
            const div = document.createElement('div');
            div.className = 'message assistant';
            div.innerHTML = `<div class="message-content"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return div;
        }
        function removeTypingIndicator(el) { if (el && el.remove) el.remove(); }
        function addMessage(role, content) {
            const container = document.getElementById('chatMessages');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `<div class="message-content">${content}</div>`;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        async function clientLogout() { await fetch('/api/client/logout', { method: 'POST', credentials: 'include' }); location.reload(); }
    </script>
</body>
</html>
'''

# ---------- COMPANY CHAT HTML (Direct Link) ----------
COMPANY_CHAT_HTML = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ company_name }} - Q&A Bot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header {
            background: white;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
            text-align: center;
        }
        .header h1 { color: #667eea; }
        .chat-container {
            background: white;
            border-radius: 10px;
            height: 600px;
            display: flex;
            flex-direction: column;
        }
        .chat-header {
            padding: 15px;
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 10px 10px 0 0;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        .message { margin-bottom: 15px; display: flex; }
        .message.user { justify-content: flex-end; }
        .message-content {
            max-width: 70%;
            padding: 10px 15px;
            border-radius: 10px;
        }
        .message.user .message-content { background: #667eea; color: white; }
        .message.assistant .message-content { background: #f0f0f0; color: #333; }
        .chat-input {
            padding: 20px;
            border-top: 1px solid #ddd;
            display: flex;
            gap: 10px;
        }
        .chat-input input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .chat-input button {
            padding: 10px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            width: auto;
        }
        .sources {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
            border-top: 1px solid #eee;
            padding-top: 5px;
        }
        .view-docs-btn {
            background: #28a745;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
        }
        .typing-indicator {
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 10px 15px;
        }
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background-color: #999;
            border-radius: 50%;
            animation: bounce 1.4s infinite;
        }
        @keyframes bounce { 0%,60%,100% { transform: translateY(0); } 30% { transform: translateY(-10px); } }
        .modal {
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.7);
        }
        .modal-content {
            background-color: white;
            margin: 5% auto;
            padding: 20px;
            width: 80%;
            height: 80%;
            border-radius: 10px;
            overflow-y: auto;
            position: relative;
        }
        .close-modal {
            position: absolute;
            right: 20px;
            top: 10px;
            font-size: 28px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏦 {{ company_name }}</h1>
            <p>Ask questions about our policies and documents</p>
        </div>
        <div class="chat-container">
            <div class="chat-header">
                <strong>Customer Support</strong>
                <button class="view-docs-btn" onclick="viewDocuments()">📄 View Documents</button>
            </div>
            <div class="chat-messages" id="chatMessages">
                <div class="message assistant"><div class="message-content">Hello! How can I help you today?</div></div>
            </div>
            <div class="chat-input">
                <input type="text" id="questionInput" placeholder="Type your question..." onkeypress="if(event.key==='Enter') askQuestion()">
                <button onclick="askQuestion()">Send</button>
            </div>
        </div>
    </div>
    <div id="docModal" class="modal">
        <div class="modal-content">
            <span class="close-modal" onclick="closeDocModal()">&times;</span>
            <h3 id="modalTitle">Documents</h3>
            <div id="modalContent"></div>
        </div>
    </div>
    <script>
        let sessionId = Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        let companyUrl = '{{ company_url }}', companyName = '{{ company_name }}';
        async function viewDocuments() {
            const response = await fetch(`/api/company/${companyUrl}/documents`);
            const docs = await response.json();
            let html = '<h4>Available Documents (Read Only)</h4><ul>';
            for (const doc of docs) html += `<li><strong>${doc.filename}</strong> - <button onclick="viewDocumentContent('${doc.filename}')">View</button></li>`;
            html += '</ul>';
            document.getElementById('modalTitle').innerHTML = `${companyName} - Documents`;
            document.getElementById('modalContent').innerHTML = html;
            document.getElementById('docModal').style.display = 'block';
        }
        async function viewDocumentContent(filename) {
            const response = await fetch(`/api/company/${companyUrl}/document/${encodeURIComponent(filename)}`);
            const content = await response.text();
            document.getElementById('modalContent').innerHTML = `<pre style="white-space: pre-wrap;">${escapeHtml(content)}</pre>`;
        }
        function escapeHtml(text) { const div = document.createElement('div'); div.textContent = text; return div.innerHTML; }
        function closeDocModal() { document.getElementById('docModal').style.display = 'none'; }
        async function askQuestion() {
            const input = document.getElementById('questionInput');
            const question = input.value.trim();
            if (!question) return;
            input.value = '';
            addMessage('user', question);
            const typingId = addTypingIndicator();
            try {
                const response = await fetch('/api/chat/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ company_name: companyName, session_id: sessionId, question: question })
                });
                const result = await response.json();
                removeTypingIndicator(typingId);
                if (response.ok) {
                    let answer = result.answer;
                    if (result.sources && result.sources.length) answer += `<div class="sources">📚 Sources: ${result.sources.join(', ')}</div>`;
                    addMessage('assistant', answer);
                } else { addMessage('assistant', `Error: ${result.error}`); }
            } catch(err) { removeTypingIndicator(typingId); addMessage('assistant', 'Error: Cannot reach server. Is Ollama running?'); }
        }
        function addTypingIndicator() {
            const container = document.getElementById('chatMessages');
            const div = document.createElement('div');
            div.className = 'message assistant';
            div.innerHTML = `<div class="message-content"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            return div;
        }
        function removeTypingIndicator(el) { if (el && el.remove) el.remove(); }
        function addMessage(role, content) {
            const container = document.getElementById('chatMessages');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `<div class="message-content">${content}</div>`;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
    </script>
</body>
</html>
'''

# ---------- FLASK ROUTES ----------
@flask_app.route('/company')
def company_portal():
    return render_template_string(COMPANY_PORTAL_HTML)

@flask_app.route('/')
def home():
    return redirect(url_for('company_portal'))

@flask_app.route('/client')
def client_portal():
    return render_template_string(CLIENT_PORTAL_HTML)

@flask_app.route('/company/<company_url>')
def company_chat(company_url):
    companies = load_companies()
    for name, data in companies.items():
        if data.get('company_url') == company_url and data.get('documents_loaded'):
            return render_template_string(COMPANY_CHAT_HTML, company_name=name, company_url=company_url)
    return "Company not found or documents not loaded", 404

@flask_app.route('/api/company/<company_url>/documents')
def get_company_documents(company_url):
    companies = load_companies()
    company_name = None
    for name, data in companies.items():
        if data.get('company_url') == company_url:
            company_name = name
            break
    if not company_name:
        return jsonify([]), 404
    company_dir = COMPANIES_DIR / company_name
    docs = []
    for file_path in company_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.docx', '.txt']:
            docs.append({"filename": file_path.name, "pages": 0, "type": file_path.suffix[1:]})
    return jsonify(docs)

@flask_app.route('/api/company/<company_url>/document/<filename>')
def get_document_content(company_url, filename):
    companies = load_companies()
    company_name = None
    for name, data in companies.items():
        if data.get('company_url') == company_url:
            company_name = name
            break
    if not company_name:
        return "Company not found", 404
    file_path = COMPANIES_DIR / company_name / filename
    if not file_path.exists():
        return "File not found", 404
    if filename.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}
    ext = filename.split('.')[-1].lower()
    if ext == 'pdf':
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        content = '\n\n'.join([p.page_content for p in pages])
    elif ext == 'docx':
        loader = Docx2txtLoader(str(file_path))
        pages = loader.load()
        content = '\n\n'.join([p.page_content for p in pages])
    else:
        content = "Preview not available for this file type."
    return content, 200, {'Content-Type': 'text/plain; charset=utf-8'}

@flask_app.route('/api/client/register', methods=['POST'])
def register_client():
    data = request.json
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    clients = load_clients()
    if email in clients:
        return jsonify({"error": "Email already registered"}), 400
    clients[email] = {
        "name": name,
        "email": email,
        "password_hash": generate_password_hash(password),
        "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_clients(clients)
    return jsonify({"message": "Registration successful"}), 200

@flask_app.route('/api/client/login', methods=['POST'])
def login_client():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    clients = load_clients()
    if email not in clients:
        return jsonify({"error": "Invalid credentials"}), 401
    if not check_password_hash(clients[email]['password_hash'], password):
        return jsonify({"error": "Invalid credentials"}), 401
    session['client_email'] = email
    session['client_name'] = clients[email]['name']
    session.permanent = True
    return jsonify({"success": True, "client": {"name": clients[email]['name'], "email": email}}), 200

@flask_app.route('/api/client/logout', methods=['POST'])
def logout_client():
    session.pop('client_email', None)
    session.pop('client_name', None)
    return jsonify({"success": True}), 200

@flask_app.route('/api/company/register', methods=['POST'])
def register_company():
    data = request.json
    company_name = data.get('company_name')
    email = data.get('email')
    password = data.get('password')
    chat_history_enabled = data.get('chat_history_enabled', False)
    companies = load_companies()
    if company_name in companies:
        return jsonify({"error": "Company already exists"}), 400
    for v in companies.values():
        if v.get('email') == email:
            return jsonify({"error": "Email already registered"}), 400
    company_url = company_name.lower().replace(' ', '_').replace('-', '_')
    companies[company_name] = {
        "company_name": company_name,
        "email": email,
        "password_hash": generate_password_hash(password),
        "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "documents_loaded": False,
        "chat_history_enabled": chat_history_enabled,
        "company_url": company_url
    }
    save_companies(companies)
    (COMPANIES_DIR / company_name).mkdir(exist_ok=True)
    return jsonify({"message": "Registration successful", "company_url": company_url}), 200

@flask_app.route('/api/company/login', methods=['POST'])
def login_company():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    companies = load_companies()
    for name, comp in companies.items():
        if comp.get('email') == email and check_password_hash(comp.get('password_hash'), password):
            session['company_name'] = name
            session.permanent = True
            return jsonify({"success": True, "company": {
                "company_name": name,
                "email": comp['email'],
                "documents_loaded": comp.get('documents_loaded', False),
                "chat_history_enabled": comp.get('chat_history_enabled', False),
                "company_url": comp.get('company_url')
            }}), 200
    return jsonify({"error": "Invalid credentials"}), 401

@flask_app.route('/api/company/logout', methods=['POST'])
def logout_company():
    session.pop('company_name', None)
    return jsonify({"success": True}), 200

@flask_app.route('/api/company/upload', methods=['POST'])
def upload_documents():
    if 'company_name' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    company_name = session['company_name']
    files = request.files.getlist('documents')
    if len(files) > 5:
        return jsonify({"error": "Max 5 files"}), 400
    company_dir = COMPANIES_DIR / company_name
    for f in files:
        if f.filename:
            f.save(str(company_dir / f.filename))
    return jsonify({"message": f"Uploaded {len(files)} files"}), 200

@flask_app.route('/api/company/process', methods=['POST'])
def process_documents_route():
    if 'company_name' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    company_name = session['company_name']
    company_dir = COMPANIES_DIR / company_name
    uploaded = []
    for p in company_dir.iterdir():
        if p.is_file() and p.suffix.lower() in ['.pdf', '.docx', '.txt']:
            with open(p, 'rb') as f:
                uploaded.append({'name': p.name, 'content': f.read()})
    if not uploaded:
        return jsonify({"error": "No documents found"}), 400
    try:
        vs, stats, cnt = process_documents_for_company(company_name, uploaded)
        company_retrievers[company_name] = OptimizedRetriever(vs)
        companies = load_companies()
        companies[company_name]['documents_loaded'] = True
        save_companies(companies)
        return jsonify({"message": "Processed", "company_url": companies[company_name]['company_url'], "chunks": cnt}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route('/api/company/chat_history', methods=['GET'])
def get_chat_history():
    if 'company_name' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    company_name = session['company_name']
    ch = load_chat_history()
    company_history = ch.get(company_name, [])
    fixed_history = []
    for sess in company_history:
        if isinstance(sess, dict):
            if 'question' in sess and 'answer' in sess:
                new_sess = {
                    "session_id": sess.get('session_id', str(uuid.uuid4())),
                    "client_email": "anonymous",
                    "client_name": "Guest",
                    "start_time": sess.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "messages": [{
                        "question": sess['question'],
                        "answer": sess['answer'],
                        "timestamp": sess.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        "sources": sess.get('sources', '')
                    }]
                }
                fixed_history.append(new_sess)
            else:
                if not isinstance(sess.get('messages'), list):
                    sess['messages'] = []
                fixed_history.append(sess)
        else:
            fixed_history.append(sess)
    return jsonify({"history": fixed_history}), 200

@flask_app.route('/api/companies', methods=['GET'])
def list_companies():
    companies = load_companies()
    available = [{"company_name": n, "company_url": d['company_url']} for n, d in companies.items() if d.get('documents_loaded')]
    return jsonify(available), 200

@flask_app.route('/api/chat/ask', methods=['POST'])
def ask_question():
    data = request.json
    company_name = data.get('company_name')
    session_id = data.get('session_id')
    question = data.get('question')
    
    client_email = session.get('client_email', 'anonymous')
    client_name = session.get('client_name', 'Guest')
    
    if company_name not in company_retrievers:
        return jsonify({"error": "Documents not loaded yet"}), 404
    retriever = company_retrievers[company_name]
    context, sources, _ = retriever.get_comprehensive_context(question)
    if not context:
        answer = "I cannot find that information in the provided documents."
    else:
        llm = get_llm(temperature=0.1)
        prompt = ChatPromptTemplate.from_template(HUMAN_PROMPT)
        answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    
    companies = load_companies()
    if companies.get(company_name, {}).get('chat_history_enabled'):
        ch = load_chat_history()
        if company_name not in ch:
            ch[company_name] = []
        
        existing_session = None
        for sess in ch[company_name]:
            if sess.get('session_id') == session_id:
                existing_session = sess
                break
        
        if existing_session:
            if not isinstance(existing_session.get('messages'), list):
                existing_session['messages'] = []
            existing_session['messages'].append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sources": ", ".join(sources)
            })
        else:
            ch[company_name].append({
                "session_id": session_id,
                "client_email": client_email,
                "client_name": client_name,
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "messages": [{
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sources": ", ".join(sources)
                }]
            })
        save_chat_history(ch)
    
    return jsonify({"answer": answer, "sources": sources}), 200

# ==================== STREAMLIT UI ====================
def run_flask():
    flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

def main():
    st.set_page_config(page_title="BFSI Multi-Tenant Platform", page_icon="🏦", layout="wide")
    st.markdown("""
    <style>
        .main-header { background: linear-gradient(135deg, #0A4D8C 0%, #1E88E5 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; color: white; text-align: center; }
        .platform-card { background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center; margin: 1rem 0; height: 100%; }
        .platform-button { background: linear-gradient(135deg, #0A4D8C 0%, #1E88E5 100%); color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; font-weight: 600; margin-top: 1rem; display: inline-block; text-decoration: none; }
        .platform-button:hover { opacity: 0.9; transform: translateY(-2px); }
    </style>
    <div class="main-header"><h1>🏦 BFSI Multi-Tenant Document Assistant</h1><p>Client Login | Company Chat History | Semantic Chunking | Full Document Viewer</p></div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="platform-card"><h2>🏢 Company Portal</h2><p>Register, upload documents, view chat history with client details</p><a href="http://localhost:5000/company" target="_blank" class="platform-button">Open Company Portal</a></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="platform-card"><h2>👥 Client Portal</h2><p>Register/Login, ask questions to companies, view documents</p><a href="http://localhost:5000/client" target="_blank" class="platform-button">Open Client Portal</a></div>', unsafe_allow_html=True)
    st.info("📌 **Setup**: This app uses a hosted LLM via OpenRouter. Set `OPENROUTER_API_KEY` in the app's secrets. Embeddings run locally on CPU (no key needed).")

def _startup_init():
    try:
        migrate_old_chat_history()
        load_existing_retrievers()
    except Exception as _e:
        print(f"Startup init warning: {_e}")


# When served by gunicorn (cloud) the module is imported, not run via
# `streamlit run`, so __name__ != "__main__". Initialise state on import.
if __name__ != "__main__":
    _startup_init()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Migrating old chat history if needed...")
    _startup_init()
    print("Starting Flask and Streamlit...")
    print("="*60 + "\n")
    threading.Thread(target=run_flask, daemon=True).start()
    time.sleep(2)
    webbrowser.open("http://localhost:5000/company")
    webbrowser.open("http://localhost:5000/client")
    main()