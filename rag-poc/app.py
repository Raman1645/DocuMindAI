"""
app.py - Streamlit Interactive Frontend UI for RAG Document Q&A Assistant.

Integrates file uploading, document chunking, ChromaDB vector indexing,
BM25 sparse search, Cross-Encoder reranking, and citation-grounded LLM generation.
"""

import os
import shutil
from pathlib import Path
import streamlit as st

from config import (
    DATA_DIR,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    RERANKER_MODEL_NAME,
    DEFAULT_LLM_MODEL,
)
from ingestion import ingest_file
from embeddings import add_documents_to_vector_store, get_vector_store
from retrieval import build_bm25_index, retrieve_and_rerank
import importlib
import generation
importlib.reload(generation)
from generation import generate_answer

# Page Configuration
st.set_page_config(
    page_title="RAG Document Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for polished presentation
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 0px 6px 6px 0px;
    }
    .source-tag {
        font-weight: 600;
        color: #1D4ED8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initializes Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "indexed_chunks" not in st.session_state:
        st.session_state.indexed_chunks = []
    if "bm25_index" not in st.session_state:
        st.session_state.bm25_index = None
    if "indexed_filenames" not in st.session_state:
        st.session_state.indexed_filenames = []


def process_uploaded_files(uploaded_files):
    """Saves uploaded files, ingests, embeds, and updates BM25 + Chroma indices."""
    new_chunks = []
    new_filenames = []
    
    for uploaded_file in uploaded_files:
        save_path = DATA_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        chunks = ingest_file(save_path)
        new_chunks.extend(chunks)
        new_filenames.append(uploaded_file.name)
        
    if new_chunks:
        # Update Chroma Vector Database
        add_documents_to_vector_store(new_chunks)
        
        # Combine all chunks and rebuild BM25 index
        st.session_state.indexed_chunks.extend(new_chunks)
        st.session_state.bm25_index, _ = build_bm25_index(st.session_state.indexed_chunks)
        st.session_state.indexed_filenames.extend(new_filenames)
        return len(new_chunks)
    return 0


def load_default_sample_data():
    """Loads sample files from sample_data/ if no user files are uploaded yet."""
    sample_dir = Path(__file__).parent / "sample_data"
    sample_txt = sample_dir / "company_policy.txt"
    sample_pdf = sample_dir / "employee_handbook.pdf"
    
    chunks = []
    filenames = []
    
    if sample_txt.exists():
        chunks.extend(ingest_file(sample_txt))
        filenames.append(sample_txt.name)
    if sample_pdf.exists():
        chunks.extend(ingest_file(sample_pdf))
        filenames.append(sample_pdf.name)
        
    if chunks:
        add_documents_to_vector_store(chunks)
        st.session_state.indexed_chunks = chunks
        st.session_state.bm25_index, _ = build_bm25_index(chunks)
        st.session_state.indexed_filenames = filenames
        return len(chunks)
    return 0


def clear_index_and_chat():
    """Clears local Chroma DB, indexed chunks, and conversation history."""
    if Path(CHROMA_PERSIST_DIR).exists():
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR)
        except Exception:
            pass
    st.session_state.messages = []
    st.session_state.indexed_chunks = []
    st.session_state.bm25_index = None
    st.session_state.indexed_filenames = []


def main():
    initialize_session_state()
    
    # Sidebar Dashboard Controls
    with st.sidebar:
        st.header("⚙️ Pipeline Dashboard")
        st.subheader("1. Document Ingestion")
        
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT Documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload custom documents to index into ChromaDB & BM25.",
        )
        
        if st.button("📥 Process & Index Uploads", use_container_width=True):
            if uploaded_files:
                with st.spinner("Chunking & Embedding documents..."):
                    count = process_uploaded_files(uploaded_files)
                    st.success(f"Indexed {count} new text chunks!")
            else:
                st.warning("Please select at least one PDF or TXT file to upload.")
                
        if not st.session_state.indexed_chunks:
            if st.button("🚀 Load Sample Policy & Handbook", use_container_width=True):
                with st.spinner("Loading demo sample documents..."):
                    count = load_default_sample_data()
                    if count > 0:
                        st.success(f"Indexed {count} sample document chunks!")
                    else:
                        st.error("Sample files not found. Please upload a file above.")
                        
        st.divider()
        st.subheader("📊 Indexing Status")
        st.write(f"**Indexed Documents:** {len(set(st.session_state.indexed_filenames))}")
        if st.session_state.indexed_filenames:
            for fname in set(st.session_state.indexed_filenames):
                st.caption(f"• {fname}")
        st.write(f"**Total Vector Chunks:** {len(st.session_state.indexed_chunks)}")
        
        st.divider()
        st.subheader("🛠️ Tech Stack & Models")
        st.caption(f"**Embeddings:** `{EMBEDDING_MODEL_NAME}`")
        st.caption(f"**Reranker:** `{RERANKER_MODEL_NAME}`")
        st.caption(f"**LLM:** `{os.getenv('LLM_MODEL_NAME', DEFAULT_LLM_MODEL)}`")
        st.caption(f"**Chunk Size / Overlap:** {CHUNK_SIZE} / {CHUNK_OVERLAP}")
        
        st.divider()
        if st.button("🗑️ Reset Vector Database & History", type="secondary", use_container_width=True):
            clear_index_and_chat()
            st.rerun()

    # Main Chat Interface Header
    st.markdown('<div class="main-header">📄 RAG Document Q&A Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask questions grounded in your uploaded documents. Powered by Hybrid Retrieval (Chroma + BM25) and Cross-Encoder Reranking.</div>',
        unsafe_allow_html=True,
    )

    # Render previous conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 Supporting Source Chunks (Citations)", expanded=False):
                    for src in message["sources"]:
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <span class="source-tag">{src['source_tag']}</span> <b>{src['file_name']}</b> (Page {src['page']}) — <i>Chunk ID: {src['chunk_id']}</i><br>
                                <small><b>Relevance Score:</b> {src['score']:.4f}</small><br>
                                <div style="margin-top: 4px; font-size: 0.9rem;">{src['content']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # User Question Input Box
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            if not st.session_state.indexed_chunks:
                warning_msg = (
                    "⚠️ No documents are currently indexed in the vector database! "
                    "Please upload a document or click **'Load Sample Policy & Handbook'** in the sidebar."
                )
                st.warning(warning_msg)
                st.session_state.messages.append({"role": "assistant", "content": warning_msg})
            else:
                with st.spinner("Searching & Reranking context chunks..."):
                    # 1. Complete Hybrid Retrieval + Reranking Pipeline
                    retrieved_chunks = retrieve_and_rerank(
                        query=user_query,
                        documents=st.session_state.indexed_chunks,
                        bm25_index=st.session_state.bm25_index,
                    )
                    
                    # 2. LLM Generation Chain with Conversational Memory
                    previous_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else []
                    response = generate_answer(user_query, retrieved_chunks, chat_history=previous_history)
                    answer_text = response["answer"]
                    sources = response["sources"]

                    # Display Answer
                    st.write(answer_text)

                    # Display Expandable Sources (collapsed by default)
                    if sources:
                        with st.expander("📚 Supporting Source Chunks (Citations)", expanded=False):
                            for src in sources:
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <span class="source-tag">{src['source_tag']}</span> <b>{src['file_name']}</b> (Page {src['page']}) — <i>Chunk ID: {src['chunk_id']}</i><br>
                                        <small><b>Cross-Encoder Score:</b> {src['score']:.4f}</small><br>
                                        <div style="margin-top: 4px; font-size: 0.9rem;">{src['content']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Append Assistant response to history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer_text, "sources": sources}
                    )


if __name__ == "__main__":
    main()
