# 🧠 DocuMindAI — Enterprise Advanced RAG System

[![CI Tests](https://github.com/Raman1645/DocuMindAI/actions/workflows/ci.yml/badge.svg)](https://github.com/Raman1645/DocuMindAI/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**DocuMindAI** is a production-grade, experimentally validated **Evidence-Grounded Document Intelligence & Advanced RAG System**. It integrates dense vector search, BM25 sparse search with Reciprocal Rank Fusion (RRF), Cross-Encoder reranking, multi-model generation (Google Gemini, Groq, Ollama), conversational query rewriting, confidence-aware abstention, and citation verification.

---

## 🏗️ System Architecture

```text
                                  DOCUMIND AI
                                       │
                                Document Upload
                                       │
                             ┌─────────▼─────────┐
                             │ SHA-256 Hashing   │ (Deduplication Check)
                             └─────────┬─────────┘
                                       │
                              Chunk + Metadata
                                       │
                        ┌──────────────┴──────────────┐
                        │                             │
                 Dense Embeddings                   BM25
              (BAAI/bge-small-en-v1.5)        (BM25Okapi Sparse)
                        │                             │
                    Chroma DB                         │
                        └──────────────┬──────────────┘
                                       │
                            Reciprocal Rank Fusion
                                   (k = 60)
                                       │
                             Cross-Encoder Reranker
                         (ms-marco-MiniLM-L-6-v2)
                                       │
                              Confidence Check
                                 /          \
                       (Score < τ)          (Score ≥ τ)
                              │                  │
                           Abstain               ↓
                         ("Don't know")   Conversational Rewriter
                                                 ↓
                                       Pluggable LLM Factory
                                ┌────────────────┼────────────────┐
                                ↓                ↓                ↓
                             Gemini             Groq            Ollama
                        (gemini-3.6-flash) (Llama-3.1-8B)      (Local)
                                └────────────────┼────────────────┘
                                                 ↓
                                         Answer Generation
                                                 ↓
                                      Citation Verifier Layer
                                                 ↓
                                         Answer + Citations
                                                 │
                                     Observability Telemetry
```

---

## 📊 Empirical Evaluation & Benchmark Results

DocuMindAI was benchmarked across a version-controlled **80-question evaluation dataset** spanning 7 distinct query categories + unanswerables.

### 1. Retrieval Pipeline Ablation Study ($K = 4$)

| Retrieval Configuration | Hit Rate@4 | Recall@4 | Precision@4 | MRR@4 | Context Precision | Avg Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dense-Only** (`BGE-small` + `Chroma`) | 97.1% | 93.9% | 26.8% | 0.9131 | 0.2497 | **26.8 ms** |
| **BM25-Only** (`BM25Okapi` Sparse) | 95.7% | 92.8% | 26.4% | 0.9119 | 0.2506 | **0.2 ms** |
| **Hybrid** (`Dense + BM25 + RRF`) | 98.6% | 95.6% | 27.5% | 0.9226 | 0.2551 | **26.7 ms** |
| **Hybrid + Reranker** (`Cross-Encoder`) | **100.0%** | **96.3%** | **27.9%** | **0.9440** | **0.2634** | **244.2 ms** |

> **Key Takeaways**:
> - Fusing Dense and BM25 with RRF increased Recall by **+1.7%** over dense-only and **+2.8%** over BM25-only.
> - Adding the Cross-Encoder Reranker achieved **100.0% Hit Rate**, pushed Recall to **96.3%**, and boosted MRR to **0.9440** (placing the ground-truth chunk at Rank 1 in almost all queries).

---

### 2. Document Chunking Strategy Ablation ($K = 4$)

| Chunking Configuration | Total Chunks | Recall@4 | Precision@4 | MRR@4 | Context Precision | Avg Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Small** (300 chars, overlap 50) | 32 | 94.7% | 27.9% | 0.9024 | 0.2580 | **209.6 ms** |
| **Baseline** (600 chars, overlap 100) | 17 | **96.3%** | **27.9%** | **0.9440** | **0.2634** | **309.8 ms** |
| **Large** (1000 chars, overlap 150) | 8 | 96.3% | 26.8% | 0.9690 | 0.2571 | **312.9 ms** |

---

### 3. Groq LLM Production Benchmark (50-Question Stratified Ragas Subset)

| Model Name | Provider | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Abstention Acc | Generation Latency | Total Latency |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **GPT-OSS 20B (`openai/gpt-oss-20b`)** 🏆 | `GROQ` | **78.6%** | **78.3%** | **85.0%** | **93.8%** | **100.0%** | **4,272 ms** | **4,919 ms** |
| **GPT-OSS 120B (`openai/gpt-oss-120b`)** | `GROQ` | **75.8%** | **76.3%** | **85.0%** | **93.8%** | **100.0%** | **4,328 ms** | **4,975 ms** |
| **Qwen 3.6 27B (`qwen/qwen3.6-27b`)** | `GROQ` | **52.4%** | **46.2%** | **85.0%** | **93.8%** | **100.0%** | **19,504 ms** | **20,151 ms** |

---

### 4. `TOP_N_RERANK` Context Window Ablation Study (GPT-OSS 20B, N=50)

Systematic optimization of candidate chunks passed to the LLM generation window:

| Configuration | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Abstention Acc | Retrieval Latency | Generation Latency | Total Latency | Latency Speedup |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`TOP_N_RERANK=1`** | 72.5% | 47.5% | **88.0%** | 85.6% | **100.0%** | 611.9 ms | **1,448.3 ms** | **2,060.2 ms** | **+56.8%** |
| **`TOP_N_RERANK=2` (Optimal)** 🏆 | **81.6%** | **60.4%** | **88.0%** | 91.9% | **100.0%** | **459.3 ms** | 3,081.6 ms | **3,540.9 ms** | **+25.8%** |
| **`TOP_N_RERANK=3`** | 80.3% | 56.0% | 86.0% | 92.8% | **100.0%** | 461.8 ms | 3,742.7 ms | 4,204.5 ms | +11.8% |
| **`TOP_N_RERANK=4` (Baseline)** | 81.0% | 58.9% | 85.0% | **93.8%** | **100.0%** | 458.8 ms | 4,310.3 ms | 4,769.1 ms | Baseline |

> **Key Findings**:
> - Setting `TOP_N_RERANK = 2` removes low-relevance noise chunks, raising **Context Precision to 88.0%** and **Faithfulness to 81.6%**.
> - Token generation latency is reduced by **1,228 ms per query (25.8% faster)** while preserving **91.9% Recall** and **100.0% Abstention Accuracy**.

---

## 🛡️ Production Engineering & Guardrails

1. **Conversational Query Rewriting (`query_rewriter.py`)**:
   - Disambiguates contextual pronouns in multi-turn interactions (e.g., *"What about the second one?"* $\rightarrow$ *"What is the second leave policy?"*).
2. **Confidence-Aware Abstention Gate (`confidence.py`)**:
   - Checks Cross-Encoder logits ($\tau = -2.5$). Refuses out-of-domain queries before invoking the LLM, eliminating hallucination risk and API costs.
3. **Citation & Evidence Verification (`verifier.py`)**:
   - Post-generation verification checking that all factual claims match verified `[Source X]` citations.
4. **Idempotent Ingestion & Deduplication (`deduplication.py`)**:
   - SHA-256 document hashing skips re-indexing already uploaded files.
5. **Observability & Tracing (`observability.py`)**:
   - Millisecond-accurate telemetry tracking every pipeline stage.

---

## 🚀 Quickstart & Setup

### Local Installation

```powershell
# 1. Clone the repository
git clone https://github.com/Raman1645/DocuMindAI.git
cd DocuMindAI/rag-poc

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys in .env
# GEMINI_API_KEY=your_key_here
# GROQ_API_KEY=your_key_here

# 5. Run the Streamlit UI
streamlit run app.py
```

### Run Tests & Benchmarks

```powershell
# Run the complete test suite (31 unit & integration tests)
pytest tests/ -v

# Run the 4-way retrieval ablation study
python ablation.py

# Run the multi-model comparison benchmark
python model_benchmark.py --subset 20
```

### Docker Deployment

```powershell
docker-compose up --build
```
Access the application at `http://localhost:7860`.

---

## 💼 Resume-Ready Highlights

- **Hybrid Retrieval & Reranking Architecture**: Engineered an advanced RAG pipeline combining dense vector embeddings (`BGE-small-en-v1.5`), BM25 sparse lexical search via Reciprocal Rank Fusion ($k=60$), and HuggingFace Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`), boosting Recall@4 to **96.3%** and Hit Rate to **100.0%**.
- **Automated IR & RAG Evaluation Harness**: Built an automated benchmarking suite over an 80-question test dataset, empirically measuring Recall@K, Precision@K, MRR@K, Context Precision, and Citation Accuracy across 4 retrieval configurations.
- **Pluggable LLM Generation Factory**: Designed a provider-agnostic Model Factory supporting Google Gemini, Groq, local Ollama, and OpenAI, enabling seamless multi-model benchmarking under fixed retrieval conditions.
- **Hallucination Mitigation & Production Guardrails**: Implemented conversational query rewriting, confidence-aware score gating for deterministic refusal on unanswerable queries, SHA-256 idempotent document deduplication, and end-to-end telemetry observability.
