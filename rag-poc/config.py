"""
config.py - Configuration management for RAG Document Q&A Assistant.
Centralizes all hyperparameters, model names, directory paths, and search settings.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")

# Ensure required directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Document Chunking Settings (RecursiveCharacterTextSplitter)
# Chunk size ~600 characters (~100-150 words), overlap ~100 characters to prevent context boundary loss
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# Embedding & Vector Store Settings
# BAAI/bge-small-en-v1.5: lightweight (33M params), top-tier benchmark performance for retrieval
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_COLLECTION_NAME = "rag_documents"

# Reranker Settings
# Cross-Encoder trained on MS-MARCO, reranks (query, document) pairs for high accuracy
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval Hyperparameters
TOP_K_DENSE = 10  # Number of dense vectors to retrieve from Chroma
TOP_K_BM25 = 10   # Number of candidate docs from BM25 sparse search
TOP_N_RERANK = 2   # Empirically tuned via ablation: yields highest Faithfulness (81.6%), Context Precision (88.0%), and 26% lower latency

# LLM Generation Settings
# Supports Groq, Google Gemini, Ollama, and OpenAI via environment variables
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL_NAME", "gemini-3.6-flash")
DEFAULT_LLM_TEMPERATURE = 0.0  # Zero temperature for deterministic, factual Q&A

# Phase 4 Guardrails & Advanced RAG Settings
ENABLE_QUERY_REWRITING = True
CONFIDENCE_THRESHOLD = -2.5  # Cross-Encoder relevance score threshold for abstention
ENABLE_ANSWER_VERIFIER = True
