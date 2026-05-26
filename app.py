import streamlit as st
import os
import tempfile
import shutil
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from typing import List, Tuple
import hashlib
import time
import re
import uuid

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="BFSI Policy Assistant - Qdrant Powered",
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
    
    .performance-badge {
        background-color: #E8F5E9;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #2E7D32;
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIG ---
QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "bfsi_documents"
MODEL_NAME = "llama3"

class QdrantManager:
    """Manages Qdrant connections - no file locking issues!"""
    
    def __init__(self):
        self.vectorstore = None
        self.collection_name = COLLECTION_NAME
        
    def get_vectorstore(self, embeddings):
        """Get existing vectorstore from Qdrant"""
        try:
            # Qdrant can run in memory or persist to disk
            # Using local mode with persistence
            self.vectorstore = Qdrant.from_existing_collection(
                embedding=embeddings,
                collection_name=self.collection_name,
                path=QDRANT_PATH,  # Persists to disk
                force_recreate=False
            )
            return self.vectorstore
        except Exception as e:
            # Collection doesn't exist yet
            return None
    
    def create_vectorstore(self, documents, embeddings):
        """Create new vectorstore in Qdrant"""
        try:
            # Clean up old storage if exists
            if os.path.exists(QDRANT_PATH):
                # Qdrant handles its own file locking, no permission issues!
                shutil.rmtree(QDRANT_PATH)
                time.sleep(0.5)
            
            # Create new collection
            self.vectorstore = Qdrant.from_documents(
                documents=documents,
                embedding=embeddings,
                path=QDRANT_PATH,
                collection_name=self.collection_name,
                force_recreate=True
            )
            return self.vectorstore
        except Exception as e:
            st.error(f"Error creating Qdrant collection: {e}")
            return None

class DynamicQueryExpander:
    """Dynamically expands queries without hardcoded keywords"""
    
    def __init__(self):
        self.stop_words = {'how', 'what', 'when', 'where', 'why', 'will', 'does', 'do', 'is', 'are', 
                          'was', 'were', 'the', 'a', 'an', 'and', 'or', 'but', 'to', 'for', 'of', 
                          'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'if',
                          'then', 'else', 'so', 'be', 'been', 'being', 'have', 'has', 'having'}
    
    def extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        text_lower = text.lower()
        words = text_lower.split()
        phrases = []
        
        for i in range(len(words)):
            for length in [2, 3, 4]:
                if i + length <= len(words):
                    phrase_words = words[i:i+length]
                    meaningful_words = [w for w in phrase_words if w not in self.stop_words]
                    if len(meaningful_words) >= 2:
                        phrase = ' '.join(phrase_words)
                        phrases.append(phrase)
        
        unique_phrases = list(set(phrases))
        unique_phrases.sort(key=len, reverse=True)
        return unique_phrases[:10]
    
    def generate_query_variations(self, question: str) -> List[str]:
        """Generate multiple query variations dynamically"""
        variations = [question]
        key_phrases = self.extract_key_phrases(question)
        variations.extend(key_phrases[:5])
        
        words = question.lower().split()
        key_terms = [w for w in words if w not in self.stop_words and len(w) > 3]
        
        if len(key_terms) >= 2:
            for i in range(len(key_terms)):
                for j in range(i+1, len(key_terms)):
                    variations.append(f"{key_terms[i]} {key_terms[j]}")
            
            if len(key_terms) >= 3:
                variations.append(' '.join(key_terms[:3]))
        
        unique_variations = list(set(variations))
        return unique_variations[:10]

class DynamicRetriever:
    """Fully dynamic retriever with Qdrant's advanced search capabilities"""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.query_expander = DynamicQueryExpander()
    
    def extract_heading(self, content: str) -> str:
        """Extract heading/title from content if present"""
        lines = content.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if (len(line) < 100 and 
                (line.endswith(':') or 
                 re.match(r'^[\d\.]+\s+', line) or
                 line.isupper() or
                 any(keyword in line.lower() for keyword in ['policy', 'section', 'article', 'clause']))):
                return line
        return None
    
    def get_context(self, question: str, k: int = 12) -> Tuple[str, List[str], List[dict]]:
        """Retrieve relevant context using Qdrant's hybrid search capabilities"""
        all_docs = []
        sources_used = set()
        
        # Generate dynamic query variations
        query_variations = self.query_expander.generate_query_variations(question)
        
        # Qdrant allows multiple search strategies efficiently
        for query in query_variations[:5]:  # Limit to 5 variations for performance
            try:
                # Similarity search
                sim_docs = self.vectorstore.similarity_search(query, k=k//2)
                all_docs.extend(sim_docs)
                
                # Qdrant's MMR search for diversity
                mmr_docs = self.vectorstore.max_marginal_relevance_search(
                    query, 
                    k=k//2, 
                    fetch_k=k,
                    lambda_mult=0.7
                )
                all_docs.extend(mmr_docs)
            except Exception as e:
                continue
        
        # Broader search with original question
        try:
            broad_docs = self.vectorstore.similarity_search(question, k=k)
            all_docs.extend(broad_docs)
        except:
            pass
        
        # Remove duplicates using content hash
        unique_docs = []
        seen_content = set()
        
        for doc in all_docs:
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
                source_name = doc.metadata.get('source_name', 'Unknown')
                sources_used.add(source_name)
        
        # Dynamic relevance scoring
        scored_docs = []
        question_terms = set(question.lower().split())
        question_terms = {t for t in question_terms if t not in self.query_expander.stop_words and len(t) > 2}
        
        for doc in unique_docs:
            content = doc.page_content.lower()
            term_score = sum(content.count(term) for term in question_terms)
            length_score = min(len(content) / 500, 1.0)
            total_score = term_score + length_score
            scored_docs.append((total_score, doc))
        
        # Sort by score
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Take top documents
        top_k = min(15, len(scored_docs))
        selected_docs_with_scores = scored_docs[:top_k]
        
        # Format context cleanly
        formatted_sections = []
        doc_details = []
        
        for i, (score, doc) in enumerate(selected_docs_with_scores):
            source = doc.metadata.get('source_name', 'Unknown')
            page = doc.metadata.get('page', None)
            content = doc.page_content.strip()
            
            # Extract heading
            heading = self.extract_heading(content)
            
            # Clean section reference
            if heading:
                section_ref = f"**{heading}**"
            else:
                section_ref = f"**Section {i+1}**"
            
            # Format for context
            formatted_sections.append(
                f"{section_ref}\n"
                f"📄 Document: {source}" + (f" | Page {page}" if page else "") + "\n"
                f"{content}\n"
            )
            
            doc_details.append({
                "source": source,
                "page": page,
                "heading": heading,
                "content_preview": content[:200] + "...",
                "relevance_score": score
            })
        
        context = "\n\n---\n\n".join(formatted_sections)
        sources_list = list(sources_used)
        
        # Show debug info
        with st.expander("🔍 Retrieval Details (Powered by Qdrant)"):
            st.write(f"**Found {len(selected_docs_with_scores)} relevant sections from {len(sources_list)} document(s)**")
            st.write(f"**Search strategy:** Multi-query with {len(query_variations[:5])} variations")
            if doc_details:
                st.write("**Sources found:**")
                for detail in doc_details[:5]:
                    page_info = f", Page {detail['page']}" if detail['page'] else ""
                    heading_info = f" - {detail['heading']}" if detail['heading'] else ""
                    st.caption(f"• {detail['source']}{page_info}{heading_info} (Score: {detail['relevance_score']:.2f})")
        
        return context, sources_list, doc_details

def load_document_with_pages(file_path: str, file_type: str):
    """Load document and ensure page numbers are captured"""
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for i, doc in enumerate(docs):
            if 'page' not in doc.metadata:
                doc.metadata['page'] = i + 1
        return docs
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata['page'] = i + 1
        return docs
    else:  # txt
        loader = TextLoader(file_path)
        docs = loader.load()
        content = docs[0].page_content
        lines = content.split('\n')
        chunks = []
        chunk_size = 50
        for i in range(0, len(lines), chunk_size):
            chunk_content = '\n'.join(lines[i:i+chunk_size])
            chunk_metadata = docs[0].metadata.copy()
            chunk_metadata['page'] = f"Lines {i+1}-{min(i+chunk_size, len(lines))}"
            chunks.append(type('Document', (), {
                'page_content': chunk_content,
                'metadata': chunk_metadata
            })())
        return chunks

def process_documents(uploaded_files):
    """Process uploaded documents with proper page number extraction"""
    all_docs = []
    file_stats = []
    
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split(".")[-1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(uploaded_file.getvalue())
            path = tmp.name
        
        docs = load_document_with_pages(path, ext)
        
        for doc in docs:
            doc.metadata["source_name"] = uploaded_file.name
            doc.metadata["file_type"] = ext.upper()
            doc.metadata["doc_id"] = str(uuid.uuid4())
            doc.metadata["processed_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        all_docs.extend(docs)
        file_stats.append({
            "name": uploaded_file.name,
            "sections": len(docs),
            "type": ext.upper()
        })
        
        os.remove(path)
    
    # Intelligent chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(all_docs)
    
    # Preserve metadata
    for chunk in chunks:
        if 'page' not in chunk.metadata:
            chunk.metadata['page'] = None
    
    return chunks, file_stats, len(chunks)

# --- IMPROVED PROMPT FOR CLEAN RESPONSES ---
PROMPT_TEMPLATE = """
You are a helpful BFSI assistant that answers questions based ONLY on the provided documents.

**CRITICAL INSTRUCTIONS:**
1. Answer using ONLY the information from the context below
2. Do NOT include technical artifacts like [Section X], brackets, or special markers
3. Use clean citations: (Document Name, Page X) or (Document Name)
4. If the context includes section headings, refer to them naturally
5. Combine information from multiple sections into a complete, flowing answer
6. Use plain, professional language suitable for clients
7. If information isn't in the context, say "I cannot find this in the provided documents"

**CONTEXT:**
{context}

**QUESTION:**
{question}

**ANSWER:**
"""

# Initialize Qdrant manager
@st.cache_resource
def init_qdrant_manager():
    """Initialize Qdrant manager as cached resource"""
    return QdrantManager()

# --- MAIN UI ---
st.markdown("""
<div class="main-header">
    <h1>🏦 BFSI Document Assistant</h1>
    <p>Powered by Qdrant | Fast, Scalable, Production-Ready</p>
    <span class="performance-badge">⚡ No file locking • Concurrent users supported • Enterprise ready</span>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "retriever" not in st.session_state:
    st.session_state.retriever = None
    st.session_state.documents_loaded = False
    st.session_state.qdrant_manager = init_qdrant_manager()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.markdown("## 📚 Document Management")
    
    st.info("⚡ **Powered by Qdrant** - Enterprise-grade vector database")
    
    files = st.file_uploader(
        "Upload Documents",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'txt'],
        help="Upload policy documents, terms & conditions, or any text documents"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Load Documents", use_container_width=True):
            if files:
                with st.spinner("Processing documents with Qdrant..."):
                    chunks, file_stats, chunk_count = process_documents(files)
                    
                    if chunks:
                        embeddings = OllamaEmbeddings(model=MODEL_NAME)
                        vs = st.session_state.qdrant_manager.create_vectorstore(
                            chunks, 
                            embeddings
                        )
                        
                        if vs:
                            st.session_state.retriever = DynamicRetriever(vs)
                            st.session_state.documents_loaded = True
                            st.session_state.messages = []
                            st.success(f"✅ Loaded {len(files)} document(s) into Qdrant!")
                            
                            for stat in file_stats:
                                st.info(f"📄 {stat['name']}: {stat['sections']} sections")
                            st.success(f"📊 Created {chunk_count} searchable vectors in Qdrant")
                            time.sleep(1)
                            st.rerun()
            else:
                st.warning("Please select files first")
    
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            if os.path.exists(QDRANT_PATH):
                shutil.rmtree(QDRANT_PATH)
            st.session_state.retriever = None
            st.session_state.documents_loaded = False
            st.session_state.messages = []
            st.success("Cleared Qdrant storage!")
            st.rerun()
    
    st.markdown("---")
    
    if st.session_state.documents_loaded:
        st.success("✅ Qdrant is ready with your documents")
        st.caption("🔍 Using hybrid search (similarity + MMR)")
    else:
        st.info("📭 No documents in Qdrant yet")
    
    st.markdown("---")
    with st.expander("ℹ️ About Qdrant"):
        st.markdown("""
        **Why Qdrant?**
        - ✅ No file locking issues
        - ✅ Supports concurrent users
        - ✅ Faster similarity search
        - ✅ Production ready
        - ✅ Advanced filtering
        - ✅ Hybrid search capabilities
        
        **How it works:**
        1. Documents are vectorized and stored in Qdrant
        2. Your questions are converted to vectors
        3. Qdrant finds the most relevant sections
        4. LLM generates clean answers from the context
        """)

# Main chat area
if not st.session_state.documents_loaded:
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to the Qdrant-Powered Document Assistant!</h3>
        <p>Upload your policy documents, terms & conditions, or any text documents to get started.</p>
        <p><strong>Why Qdrant makes this better:</strong></p>
        <ul>
            <li>🚀 <strong>Faster searches</strong> - Even with hundreds of pages</li>
            <li>🔒 <strong>No file locking</strong> - Multiple users can access simultaneously</li>
            <li>🎯 <strong>Better accuracy</strong> - Advanced vector search algorithms</li>
            <li>💪 <strong>Production ready</strong> - Built for enterprise use</li>
        </ul>
        <p><strong>Example questions you can ask:</strong></p>
        <ul>
            <li>What are the key provisions in this document?</li>
            <li>What happens in case of any scenario from your documents?</li>
            <li>What are the customer responsibilities?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Sources Used"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
    
    # Chat input
    if user_input := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching Qdrant vector database..."):
                try:
                    context, sources, doc_details = st.session_state.retriever.get_context(
                        user_input, 
                        k=12
                    )
                    
                    if not context or len(doc_details) == 0:
                        st.warning("No relevant sections found in Qdrant. Please try rephrasing your question.")
                    else:
                        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                        llm = ChatOllama(model=MODEL_NAME, temperature=0)
                        
                        rag_chain = (
                            {"context": lambda x: context, "question": RunnablePassthrough()}
                            | prompt
                            | llm
                            | StrOutputParser()
                        )
                        
                        response = rag_chain.invoke(user_input)
                        st.markdown(response)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response,
                            "sources": sources
                        })
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": []
                    })

# Footer
st.markdown("---")
st.caption("⚡ Powered by Qdrant vector database | Answers generated based solely on uploaded documents | No file locking, concurrent user support")