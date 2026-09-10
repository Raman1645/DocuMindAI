"""
retrieval.py - BM25 & Reciprocal Rank Fusion (RRF) Hybrid Search with Cross-Encoder Reranking.

Combines sparse keyword search (rank_bm25) with dense vector search (ChromaDB)
via Reciprocal Rank Fusion (RRF), then applies a HuggingFace Cross-Encoder model
(cross-encoder/ms-marco-MiniLM-L-6-v2) to re-score candidates for maximum precision.
"""

import re
from typing import List, Tuple, Dict
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from config import (
    TOP_K_DENSE,
    TOP_K_BM25,
    TOP_N_RERANK,
    RERANKER_MODEL_NAME,
)
from embeddings import dense_similarity_search

# Global singleton cache for CrossEncoder model to avoid reloading weights
_RERANKER_MODEL = None


def get_reranker_model(model_name: str = RERANKER_MODEL_NAME) -> CrossEncoder:
    """
    Initializes and returns the HuggingFace CrossEncoder model singleton.
    
    Args:
        model_name: HuggingFace model identifier for cross-encoder.
        
    Returns:
        CrossEncoder: Loaded CrossEncoder model instance.
    """
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        _RERANKER_MODEL = CrossEncoder(model_name)
    return _RERANKER_MODEL


def tokenize_text(text: str) -> List[str]:
    """
    Standard alphanumeric lowercasing tokenizer for BM25 indexing and querying.
    
    Args:
        text: Raw input text string.
        
    Returns:
        List[str]: List of lowercase word tokens.
    """
    return re.findall(r"\w+", text.lower())


def build_bm25_index(documents: List[Document]) -> Tuple[BM25Okapi, List[Document]]:
    """
    Builds an in-memory BM25Okapi index over a list of chunked Document objects.
    
    Args:
        documents: List of Document objects to index.
        
    Returns:
        Tuple[BM25Okapi, List[Document]]: (BM25 index, indexed documents reference list).
    """
    tokenized_corpus = [tokenize_text(doc.page_content) for doc in documents]
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index, documents


def bm25_search(
    query: str,
    documents: List[Document],
    bm25_index: BM25Okapi,
    top_k: int = TOP_K_BM25,
) -> List[Tuple[Document, float]]:
    """
    Executes sparse BM25 keyword search over indexed documents.
    
    Args:
        query: User search query string.
        documents: Indexed document list matching the BM25 index corpus.
        bm25_index: Pre-built BM25Okapi index object.
        top_k: Maximum candidate documents to retrieve.
        
    Returns:
        List[Tuple[Document, float]]: Top-k (Document, BM25_score) pairs sorted by relevance.
    """
    if not documents or bm25_index is None:
        return []
        
    tokenized_query = tokenize_text(query)
    scores = bm25_index.get_scores(tokenized_query)
    
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda pair: pair[1], reverse=True)
    
    return doc_score_pairs[:top_k]


def reciprocal_rank_fusion(
    dense_results: List[Tuple[Document, float]],
    bm25_results: List[Tuple[Document, float]],
    rrf_k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Combines dense and sparse search rankings using Reciprocal Rank Fusion (RRF).
    Formula: RRF_Score(d) = sum( 1 / (rrf_k + rank(d)) ) across all retrieval lists.
    
    Args:
        dense_results: Ranked list of (Document, score) from dense similarity search.
        bm25_results: Ranked list of (Document, score) from BM25 sparse search.
        rrf_k: RRF constant penalty term (default 60).
        
    Returns:
        List[Tuple[Document, float]]: Merged, deduplicated documents sorted by RRF score descending.
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    
    # Process dense search rankings (1-indexed rank)
    for rank, (doc, _score) in enumerate(dense_results, start=1):
        chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        doc_map[chunk_id] = doc
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        
    # Process BM25 search rankings (1-indexed rank)
    for rank, (doc, _score) in enumerate(bm25_results, start=1):
        chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
        doc_map[chunk_id] = doc
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + (1.0 / (rrf_k + rank))
        
    # Sort unique documents by merged RRF score descending
    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    
    hybrid_results = [(doc_map[cid], rrf_scores[cid]) for cid in sorted_chunk_ids]
    return hybrid_results


def hybrid_retrieval(
    query: str,
    documents: List[Document],
    bm25_index: BM25Okapi,
    top_k_dense: int = TOP_K_DENSE,
    top_k_bm25: int = TOP_K_BM25,
    persist_directory: str = None,
    collection_name: str = None,
) -> List[Tuple[Document, float]]:
    """
    Executes hybrid dense vector + sparse BM25 retrieval with RRF fusion.
    """
    kwargs = {}
    if persist_directory:
        kwargs["persist_directory"] = persist_directory
    if collection_name:
        kwargs["collection_name"] = collection_name
        
    dense_results = dense_similarity_search(query, top_k=top_k_dense, **kwargs)
    bm25_results = bm25_search(query, documents, bm25_index, top_k=top_k_bm25)
    
    hybrid_results = reciprocal_rank_fusion(dense_results, bm25_results)
    return hybrid_results


def rerank_documents(
    query: str,
    candidate_docs: List[Document],
    top_n: int = TOP_N_RERANK,
    model_name: str = RERANKER_MODEL_NAME,
) -> List[Tuple[Document, float]]:
    """
    Reranks candidate documents using HuggingFace Cross-Encoder model (ms-marco-MiniLM-L-6-v2).
    Cross-attention between query and document text computes accurate relevance scores.
    
    Args:
        query: User search question string.
        candidate_docs: List of candidate Document objects retrieved from hybrid search.
        top_n: Number of top reranked chunks to retain.
        model_name: Model identifier for cross-encoder.
        
    Returns:
        List[Tuple[Document, float]]: Top-N (Document, cross_encoder_score) pairs sorted descending.
    """
    if not candidate_docs:
        return []
        
    reranker = get_reranker_model(model_name)
    
    # Construct [query, document_text] pairs for cross-encoder input
    pairs = [[query, doc.page_content] for doc in candidate_docs]
    
    # Predict relevance logits/scores
    scores = reranker.predict(pairs)
    
    # Combine documents with predicted cross-encoder scores
    doc_score_pairs = [(doc, float(score)) for doc, score in zip(candidate_docs, scores)]
    
    # Sort by cross-encoder score descending
    doc_score_pairs.sort(key=lambda item: item[1], reverse=True)
    
    return doc_score_pairs[:top_n]


def retrieve_and_rerank(
    query: str,
    documents: List[Document],
    bm25_index: BM25Okapi,
    top_k_dense: int = TOP_K_DENSE,
    top_k_bm25: int = TOP_K_BM25,
    top_n_rerank: int = TOP_N_RERANK,
    persist_directory: str = None,
    collection_name: str = None,
) -> List[Tuple[Document, float]]:
    """
    Complete Retrieval Pipeline:
    1. Hybrid Search (Dense Chroma + Sparse BM25 via RRF)
    2. Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    
    Args:
        query: User search question string.
        documents: List of all document chunks.
        bm25_index: Pre-built BM25 index over documents.
        top_k_dense: Number of dense vector search candidates.
        top_k_bm25: Number of BM25 search candidates.
        top_n_rerank: Final top N chunks to retain for LLM context.
        persist_directory: Optional custom Chroma path.
        collection_name: Optional custom Chroma collection name.
        
    Returns:
        List[Tuple[Document, float]]: Top-N reranked (Document, cross_encoder_score) pairs.
    """
    # 1. Hybrid Search candidate collection
    hybrid_candidates = hybrid_retrieval(
        query=query,
        documents=documents,
        bm25_index=bm25_index,
        top_k_dense=top_k_dense,
        top_k_bm25=top_k_bm25,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    
    candidate_docs = [doc for doc, _score in hybrid_candidates]
    
    # 2. Cross-Encoder Reranking
    reranked_results = rerank_documents(
        query=query,
        candidate_docs=candidate_docs,
        top_n=top_n_rerank,
    )
    
    return reranked_results


def retrieve_by_mode(
    mode: str,
    query: str,
    documents: List[Document],
    bm25_index: BM25Okapi,
    top_k: int = TOP_N_RERANK,
    top_k_dense: int = TOP_K_DENSE,
    top_k_bm25: int = TOP_K_BM25,
    persist_directory: str = None,
    collection_name: str = None,
) -> List[Tuple[Document, float]]:
    """
    Executes retrieval under a specific configuration mode for ablation testing:
    - 'dense': Dense vector similarity search only (ChromaDB + BGE-small).
    - 'bm25': BM25Okapi sparse lexical search only.
    - 'hybrid': Dense + BM25 combined via Reciprocal Rank Fusion (RRF).
    - 'hybrid_rerank': Dense + BM25 RRF + Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2).

    Args:
        mode: Retrieval strategy mode ('dense', 'bm25', 'hybrid', 'hybrid_rerank').
        query: User search question string.
        documents: List of all chunked Document objects.
        bm25_index: Pre-built BM25Okapi index.
        top_k: Final number of candidate chunks to return.
        top_k_dense: Number of dense vectors to retrieve for hybrid/dense candidates.
        top_k_bm25: Number of BM25 candidates to retrieve for hybrid/bm25.
        persist_directory: ChromaDB directory path.
        collection_name: ChromaDB collection name.

    Returns:
        List[Tuple[Document, float]]: Top-K (Document, score) pairs.
    """
    kwargs = {}
    if persist_directory:
        kwargs["persist_directory"] = persist_directory
    if collection_name:
        kwargs["collection_name"] = collection_name

    mode_lower = mode.lower().strip()

    if mode_lower == "dense":
        return dense_similarity_search(query, top_k=top_k, **kwargs)

    elif mode_lower == "bm25":
        return bm25_search(query, documents, bm25_index, top_k=top_k)

    elif mode_lower == "hybrid":
        results = hybrid_retrieval(
            query=query,
            documents=documents,
            bm25_index=bm25_index,
            top_k_dense=top_k_dense,
            top_k_bm25=top_k_bm25,
            **kwargs,
        )
        return results[:top_k]

    elif mode_lower in ("hybrid_rerank", "hybrid_reranker", "full"):
        return retrieve_and_rerank(
            query=query,
            documents=documents,
            bm25_index=bm25_index,
            top_k_dense=top_k_dense,
            top_k_bm25=top_k_bm25,
            top_n_rerank=top_k,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unsupported retrieval mode '{mode}'. Supported modes: 'dense', 'bm25', 'hybrid', 'hybrid_rerank'."
        )


if __name__ == "__main__":
    import shutil
    from ingestion import ingest_file
    from embeddings import add_documents_to_vector_store
    
    print("=== Step 4: Cross-Encoder Reranking & Pipeline Test ===")
    
    sample_dir = Path(__file__).parent / "sample_data"
    txt_file = sample_dir / "company_policy.txt"
    pdf_file = sample_dir / "employee_handbook.pdf"
    
    test_db_dir = Path("./chroma_rerank_test_db")
    try:
        if test_db_dir.exists():
            shutil.rmtree(test_db_dir)
    except PermissionError:
        pass
        
    print("\n1. Ingesting & Indexing sample documents...")
    all_chunks = []
    if txt_file.exists():
        all_chunks.extend(ingest_file(txt_file))
    if pdf_file.exists():
        all_chunks.extend(ingest_file(pdf_file))
        
    add_documents_to_vector_store(
        all_chunks, persist_directory=str(test_db_dir), collection_name="rerank_test"
    )
    
    bm25_idx, _ = build_bm25_index(all_chunks)
    print(f"Indexed {len(all_chunks)} chunks.")
    
    test_query = "What Wi-Fi network rules apply to remote workers?"
    print(f"\n2. Executing Full Pipeline (Hybrid Search + Cross-Encoder Reranker): '{test_query}'")
    
    # Step A: Hybrid Candidates before Reranking
    hybrid_candidates = hybrid_retrieval(
        test_query, all_chunks, bm25_idx, top_k_dense=4, top_k_bm25=4,
        persist_directory=str(test_db_dir), collection_name="rerank_test"
    )
    
    print("\n--- Hybrid Search Candidates (Before Reranking) ---")
    for r, (doc, rrf_score) in enumerate(hybrid_candidates, 1):
        print(f"Rank {r} [{doc.metadata['chunk_id']}] (RRF Score: {rrf_score:.5f}): {repr(doc.page_content[:90])}...")
        
    # Step B: Cross-Encoder Reranked Top Chunks
    reranked_results = retrieve_and_rerank(
        test_query, all_chunks, bm25_idx, top_k_dense=4, top_k_bm25=4, top_n_rerank=2,
        persist_directory=str(test_db_dir), collection_name="rerank_test"
    )
    
    print("\n--- Cross-Encoder Reranked Results (Final Top 2) ---")
    for r, (doc, ce_score) in enumerate(reranked_results, 1):
        print(f"Rank {r} [{doc.metadata['chunk_id']}] (Cross-Encoder Score: {ce_score:.4f}):")
        print(f"       Source: {doc.metadata['source']} (Page {doc.metadata['page']})")
        print(f"       Content Snippet: {repr(doc.page_content[:140])}...")
        
    try:
        if test_db_dir.exists():
            shutil.rmtree(test_db_dir)
    except PermissionError:
        pass
    print("\nCross-Encoder Reranking Test Completed Successfully.")
