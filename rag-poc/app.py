"""
app.py - Production Streamlit UI for DocuMindAI Advanced RAG Assistant.

Features:
- Pluggable LLM selection (Google Gemini, Groq, Local Ollama, Mock)
- 4-Way Retrieval Mode toggle (Hybrid + Reranker, Hybrid RRF, Dense-only, BM25-only)
- Guardrails: Conversational Query Rewriter, Confidence Gate, Answer Verifier
- Real-time latency & telemetry tracing via PipelineTracer
- SHA-256 Idempotent document ingestion & deduplication
"""

import os
import time
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
    CONFIDENCE_THRESHOLD,
    TOP_N_RERANK,
)
from deduplication import DocumentRegistry, compute_file_hash
from ingestion import ingest_file_idempotent, ingest_file
from embeddings import add_documents_to_vector_store, get_all_documents_from_vector_store
from retrieval import build_bm25_index, retrieve_by_mode
from query_rewriter import rewrite_query
from confidence import check_retrieval_confidence
from verifier import verify_grounded_answer
from generation import generate_answer
from observability import PipelineTracer

from llm_factory import get_available_groq_models

# Page Configuration
st.set_page_config(
    page_title="DocuMindAI - Advanced RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling for premium presentation
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 1.2rem;
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
    .telemetry-chip {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4338CA;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def hydrate_session_state_from_disk():
    """Auto-hydrates in-memory chunks and BM25 index from ChromaDB, DATA_DIR, or sample_data."""
    all_chunks = []
    filenames = []
    seen_chunk_ids = set()
    
    # 1. Pull all documents directly from persistent ChromaDB collection
    try:
        chroma_docs = get_all_documents_from_vector_store()
        for doc in chroma_docs:
            cid = doc.metadata.get("chunk_id") or str(hash(doc.page_content))
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                all_chunks.append(doc)
                src = doc.metadata.get("source") or doc.metadata.get("file_name") or "indexed_doc"
                if src not in filenames:
                    filenames.append(src)
    except Exception:
        pass

    # 2. Also check DATA_DIR and sample_data for document files
    candidate_files = []
    if DATA_DIR.exists():
        candidate_files.extend(sorted(list(DATA_DIR.glob("*.pdf")) + list(DATA_DIR.glob("*.txt"))))
    sample_dir = Path(__file__).parent / "sample_data"
    if sample_dir.exists() and not all_chunks:
        candidate_files.extend(sorted(list(sample_dir.glob("*.pdf")) + list(sample_dir.glob("*.txt"))))
        
    for fpath in candidate_files:
        try:
            chunks = ingest_file(fpath)
            for c in chunks:
                cid = c.metadata.get("chunk_id")
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    all_chunks.append(c)
            if fpath.name not in filenames:
                filenames.append(fpath.name)
        except Exception:
            pass
            
    if all_chunks:
        st.session_state.indexed_chunks = all_chunks
        st.session_state.indexed_filenames = filenames
        st.session_state.bm25_index, _ = build_bm25_index(all_chunks)


def initialize_session_state():
    """Initializes Streamlit session state variables and hydrates from disk."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "registry" not in st.session_state:
        st.session_state.registry = DocumentRegistry()
    if "indexed_chunks" not in st.session_state:
        st.session_state.indexed_chunks = []
    if "bm25_index" not in st.session_state:
        st.session_state.bm25_index = None
    if "indexed_filenames" not in st.session_state:
        st.session_state.indexed_filenames = []
        
    if not st.session_state.indexed_chunks:
        hydrate_session_state_from_disk()


def process_uploaded_files(uploaded_files):
    """Saves uploaded files and idempotently indexes new documents, ensuring session memory is hydrated."""
    new_chunks = []
    loaded_chunks = []
    new_filenames = []
    skipped_count = 0
    
    existing_chunk_ids = {c.metadata.get("chunk_id") for c in st.session_state.indexed_chunks}
    
    for uploaded_file in uploaded_files:
        save_path = DATA_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        doc_hash = compute_file_hash(save_path)
        chunks = ingest_file(save_path)
        
        if not st.session_state.registry.is_indexed(doc_hash):
            st.session_state.registry.register(
                doc_hash=doc_hash,
                file_name=uploaded_file.name,
                chunk_count=len(chunks),
            )
            new_chunks.extend(chunks)
            new_filenames.append(uploaded_file.name)
        else:
            skipped_count += 1
            
        for c in chunks:
            cid = c.metadata.get("chunk_id")
            if cid not in existing_chunk_ids:
                existing_chunk_ids.add(cid)
                loaded_chunks.append(c)
                if uploaded_file.name not in st.session_state.indexed_filenames:
                    st.session_state.indexed_filenames.append(uploaded_file.name)
            
    if new_chunks:
        add_documents_to_vector_store(new_chunks)
        
    if loaded_chunks:
        st.session_state.indexed_chunks.extend(loaded_chunks)
        st.session_state.bm25_index, _ = build_bm25_index(st.session_state.indexed_chunks)
        
    return len(new_chunks), skipped_count, len(loaded_chunks)


def load_default_sample_data():
    """Loads sample files from sample_data/ with deduplication and session hydration."""
    sample_dir = Path(__file__).parent / "sample_data"
    sample_files = sorted(list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.pdf")))
    
    total_new_chunks = 0
    loaded_chunks = []
    existing_chunk_ids = {c.metadata.get("chunk_id") for c in st.session_state.indexed_chunks}
    
    for sfile in sample_files:
        # Also copy sample files to DATA_DIR so they persist across restarts
        target_path = DATA_DIR / sfile.name
        if not target_path.exists():
            shutil.copy(sfile, target_path)
            
        doc_hash = compute_file_hash(sfile)
        chunks = ingest_file(sfile)
        
        if not st.session_state.registry.is_indexed(doc_hash):
            st.session_state.registry.register(
                doc_hash=doc_hash,
                file_name=sfile.name,
                chunk_count=len(chunks),
            )
            add_documents_to_vector_store(chunks)
            total_new_chunks += len(chunks)
            
        for c in chunks:
            cid = c.metadata.get("chunk_id")
            if cid not in existing_chunk_ids:
                existing_chunk_ids.add(cid)
                loaded_chunks.append(c)
                if sfile.name not in st.session_state.indexed_filenames:
                    st.session_state.indexed_filenames.append(sfile.name)
            
    if loaded_chunks:
        st.session_state.indexed_chunks.extend(loaded_chunks)
        st.session_state.bm25_index, _ = build_bm25_index(st.session_state.indexed_chunks)
        
    return total_new_chunks or len(loaded_chunks)


def clear_index_and_chat():
    """Clears local Chroma DB, registry, data directory, indexed chunks, and conversation history."""
    if Path(CHROMA_PERSIST_DIR).exists():
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR)
        except Exception:
            pass
    if "registry" in st.session_state:
        st.session_state.registry.clear()
        
    # Clean up uploaded files in DATA_DIR
    if DATA_DIR.exists():
        for f in DATA_DIR.glob("*"):
            if f.is_file() and f.name != "document_registry.json":
                try:
                    f.unlink()
                except Exception:
                    pass
                    
    st.session_state.messages = []
    st.session_state.indexed_chunks = []
    st.session_state.bm25_index = None
    st.session_state.indexed_filenames = []


def main():
    initialize_session_state()
    
    # -------------------------------------------------------------
    # Sidebar: Model, Retrieval, Guardrails & Ingestion Controls
    # -------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # 1. Model Provider Selection
        st.subheader("1. LLM Generation Engine")
        provider_options = ["gemini", "groq", "ollama", "mock"]
        selected_provider = st.selectbox(
            "Select Provider",
            provider_options,
            index=0 if os.getenv("GEMINI_API_KEY") else 1,
            help="Choose the active generation backend.",
        )
        
        if selected_provider == "gemini":
            model_options = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-pro", "Custom Model..."]
            sel = st.selectbox("Select Model", model_options, index=0)
            if sel == "Custom Model...":
                selected_model = st.text_input("Enter Gemini Model Identifier", value="gemini-3.6-flash")
            else:
                selected_model = sel
        elif selected_provider == "groq":
            groq_models_map = {
                "GPT-OSS 120B (openai/gpt-oss-120b)": "openai/gpt-oss-120b",
                "GPT-OSS 20B (openai/gpt-oss-20b)": "openai/gpt-oss-20b",
                "Qwen 3.6 27B (qwen/qwen3.6-27b)": "qwen/qwen3.6-27b",
            }
            sel = st.selectbox("Select Groq Model", list(groq_models_map.keys()), index=0)
            selected_model = groq_models_map[sel]
        elif selected_provider == "ollama":
            model_options = ["qwen3:8b", "qwen2.5:8b", "llama3.2:3b", "llama3.2:1b", "Custom Model..."]
            sel = st.selectbox("Select Model", model_options, index=0)
            if sel == "Custom Model...":
                selected_model = st.text_input("Enter Ollama Model Tag", value="llama3.2")
            else:
                selected_model = sel
        else:
            model_options = ["mock-deterministic"]
            selected_model = st.selectbox("Select Model", model_options, index=0)
        
        st.divider()
        
        # 2. Retrieval Strategy Selection
        st.subheader("2. Retrieval Pipeline Strategy")
        retrieval_mode_map = {
            "Hybrid + Cross-Encoder Reranker (Best)": "hybrid_rerank",
            "Hybrid (Dense + BM25 RRF)": "hybrid",
            "Dense Vector Only (Chroma + BGE)": "dense",
            "BM25 Sparse Lexical Only": "bm25",
        }
        selected_ret_label = st.selectbox(
            "Retrieval Strategy",
            list(retrieval_mode_map.keys()),
            index=0,
            help="Select the retrieval architecture strategy (validated in Phase 2 ablation).",
        )
        selected_mode = retrieval_mode_map[selected_ret_label]
        
        st.divider()
        
        # 3. Guardrails Toggles
        st.subheader("3. Quality & Guardrails")
        enable_rewriter = st.checkbox("Enable Conversational Query Rewriter", value=True)
        enable_confidence_gate = st.checkbox("Enable Confidence Abstention Gate", value=True)
        enable_verifier = st.checkbox("Enable Citation Verification Layer", value=True)
        
        st.divider()
        
        # 4. Document Ingestion
        st.subheader("4. Idempotent Ingestion")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT Documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="SHA-256 hashing skips already indexed documents.",
        )
        
        if st.button("📥 Index Uploaded Documents", use_container_width=True):
            if uploaded_files:
                with st.spinner("Processing & Hashing uploads..."):
                    new_c, skipped_c, loaded_c = process_uploaded_files(uploaded_files)
                    if new_c > 0:
                        st.success(f"✅ Indexed {new_c} new chunks! Ready for questions.")
                    elif loaded_c > 0:
                        st.success(f"✅ Loaded {loaded_c} chunks from {len(uploaded_files)} file(s)! Ready for questions.")
                    else:
                        st.success(f"✅ All {skipped_c} file(s) are active and ready! Ask your question below.")
            else:
                st.warning("Select PDF or TXT files to upload.")
                
        if not st.session_state.indexed_chunks:
            if st.button("🚀 Load Sample Policy & Handbook", use_container_width=True):
                with st.spinner("Ingesting sample documents..."):
                    count = load_default_sample_data()
                    if count > 0:
                        st.success(f"Indexed {count} chunks across sample documents!")
                    else:
                        st.info("Sample files are already loaded.")
                        
        st.divider()
        st.caption(f"**Indexed Files:** {len(set(st.session_state.indexed_filenames))}")
        st.caption(f"**Total Chunks in DB:** {len(st.session_state.indexed_chunks)}")
        
        if st.button("🗑️ Reset Vector DB & Chat", type="secondary", use_container_width=True):
            clear_index_and_chat()
            st.rerun()

    # -------------------------------------------------------------
    # Main Chat Interface
    # -------------------------------------------------------------
    st.markdown('<div class="main-header">🧠 DocuMindAI — Document Intelligence Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Advanced Hybrid RAG with Cross-Encoder Reranking, Multi-Model Factory, and Citation Guardrails.</div>',
        unsafe_allow_html=True,
    )

    # Render previous conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Render Telemetry Chips if available
            if "telemetry" in message and message["telemetry"]:
                t = message["telemetry"]
                chips_html = "".join([f'<span class="telemetry-chip">{k}: {v}</span>' for k, v in t.items()])
                st.markdown(chips_html, unsafe_allow_html=True)
                
            if "sources" in message and message["sources"]:
                with st.expander(f"📚 Supporting Source Chunks ({len(message['sources'])} citations)", expanded=False):
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

    # User Query Input
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            if not st.session_state.indexed_chunks:
                hydrate_session_state_from_disk()
                
            if not st.session_state.indexed_chunks:
                msg = "⚠️ No documents indexed yet. Please upload files or click **'Load Sample Policy & Handbook'** in the sidebar."
                st.warning(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            else:
                tracer = PipelineTracer(query=user_query)
                previous_history = st.session_state.messages[:-1] if len(st.session_state.messages) > 1 else []
                
                try:
                    # Stage 1: Query Rewriter
                    with tracer.trace_stage("query_rewriter"):
                        if enable_rewriter:
                            effective_query = rewrite_query(
                                query=user_query,
                                chat_history=previous_history,
                                provider=selected_provider,
                                model_name=selected_model,
                            )
                        else:
                            effective_query = user_query
                            
                    if effective_query != user_query:
                        st.caption(f"🔍 *Rewritten Query:* `{effective_query}`")

                    # Stage 2: Retrieval
                    with tracer.trace_stage("retrieval"):
                        retrieved_chunks = retrieve_by_mode(
                            mode=selected_mode,
                            query=effective_query,
                            documents=st.session_state.indexed_chunks,
                            bm25_index=st.session_state.bm25_index,
                            top_k=TOP_N_RERANK,
                        )

                    # Stage 3: Confidence Check
                    is_confident = True
                    max_logit = 0.0
                    if enable_confidence_gate and selected_mode == "hybrid_rerank":
                        is_confident, max_logit, _abstain_msg = check_retrieval_confidence(
                            retrieved_chunks, threshold=CONFIDENCE_THRESHOLD
                         )

                    # Stage 4: Generation
                    with tracer.trace_stage("generation"):
                        if not is_confident:
                            answer_text = "I don't know based on the provided documents."
                            sources = []
                            verification = {"is_verified": True, "status": "abstained_low_confidence"}
                        else:
                            response = generate_answer(
                                query=effective_query,
                                retrieved_chunks=retrieved_chunks,
                                chat_history=previous_history,
                                model_name=selected_model,
                                provider=selected_provider,
                                max_tokens=1024,
                            )
                            answer_text = response["answer"]
                            sources = response["sources"]
                            verification = response.get("verification", {})

                    # Display Answer
                    st.write(answer_text)

                    # Display Telemetry Summary
                    summary = tracer.get_summary()
                    telemetry = {
                        "Total Latency": f"{summary['total_latency_ms']:.0f}ms",
                        "Retrieval": f"{summary['stages_ms'].get('retrieval', 0):.0f}ms",
                        "Generation": f"{summary['stages_ms'].get('generation', 0):.0f}ms",
                        "Model": f"{selected_provider.upper()}/{selected_model}",
                        "Verified": "✅ YES" if verification.get("is_verified", False) else "⚠️ Refusal/Unverified",
                    }
                    
                    chips_html = "".join([f'<span class="telemetry-chip">{k}: {v}</span>' for k, v in telemetry.items()])
                    st.markdown(chips_html, unsafe_allow_html=True)

                    # Display Sources Expandable
                    if sources:
                        with st.expander(f"📚 Supporting Source Chunks ({len(sources)} citations)", expanded=False):
                            for src in sources:
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <span class="source-tag">{src['source_tag']}</span> <b>{src['file_name']}</b> (Page {src['page']}) — <i>Chunk ID: {src['chunk_id']}</i><br>
                                        <small><b>Score:</b> {src['score']:.4f}</small><br>
                                        <div style="margin-top: 4px; font-size: 0.9rem;">{src['content']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Append to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer_text,
                        "sources": sources,
                        "telemetry": telemetry,
                    })
                except Exception as e:
                    fallback_err = "⚠️ **Unable to process your request at this moment.** Please try again in a few seconds or switch model providers in the sidebar."
                    st.warning(fallback_err)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": fallback_err,
                        "sources": [],
                    })


if __name__ == "__main__":
    main()
