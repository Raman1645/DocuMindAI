"""
test_ablation.py - Unit and Integration Tests for Phase 2 Retrieval Ablation Suite.

Validates that:
1. retrieve_by_mode properly executes across all 4 modes ('dense', 'bm25', 'hybrid', 'hybrid_rerank').
2. Invalid mode strings raise ValueError.
3. Custom chunking granularities (small, baseline, large) generate expected document slices.
"""

import pytest
import shutil
from pathlib import Path
from langchain_core.documents import Document

from ingestion import chunk_documents
from embeddings import add_documents_to_vector_store
from retrieval import build_bm25_index, retrieve_by_mode


@pytest.fixture(scope="module")
def sample_corpus(tmp_path_factory):
    """Creates a temporary in-memory Chroma DB and BM25 index for ablation tests."""
    temp_dir = tmp_path_factory.mktemp("chroma_test_ablation")
    
    docs = [
        Document(
            page_content="ACME Corp full-time employees are eligible for a one-time home office setup stipend of $500.",
            metadata={"source": "policy.txt", "page": 1, "chunk_id": "policy_p1_c0"},
        ),
        Document(
            page_content="Full-time employees receive 20 days of Paid Time Off (PTO) per calendar year, accrued monthly at 1.66 days.",
            metadata={"source": "benefits.pdf", "page": 2, "chunk_id": "benefits_p2_c1"},
        ),
        Document(
            page_content="All production code changes require a minimum of two approving peer reviews before merging into main.",
            metadata={"source": "engineering.txt", "page": 1, "chunk_id": "eng_p1_c0"},
        ),
        Document(
            page_content="All laptops must have FileVault or BitLocker full-disk encryption enabled.",
            metadata={"source": "policy.txt", "page": 1, "chunk_id": "policy_p1_c1"},
        ),
    ]
    
    add_documents_to_vector_store(
        docs,
        persist_directory=str(temp_dir),
        collection_name="test_ablation_col",
    )
    bm25_idx, _ = build_bm25_index(docs)
    
    yield {
        "docs": docs,
        "bm25_idx": bm25_idx,
        "persist_dir": str(temp_dir),
        "collection_name": "test_ablation_col",
    }
    
    try:
        shutil.rmtree(temp_dir)
    except PermissionError:
        pass


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_retrieve_by_mode_valid(sample_corpus, mode):
    """Tests that all 4 retrieval modes return top-k documents with valid scores."""
    query = "What is the home office setup stipend amount?"
    results = retrieve_by_mode(
        mode=mode,
        query=query,
        documents=sample_corpus["docs"],
        bm25_index=sample_corpus["bm25_idx"],
        top_k=2,
        persist_directory=sample_corpus["persist_dir"],
        collection_name=sample_corpus["collection_name"],
    )
    
    assert len(results) <= 2
    assert len(results) > 0
    doc, score = results[0]
    assert isinstance(doc, Document)
    assert isinstance(score, (int, float))
    assert "chunk_id" in doc.metadata


def test_retrieve_by_mode_invalid(sample_corpus):
    """Tests that unsupported mode raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported retrieval mode"):
        retrieve_by_mode(
            mode="unsupported_quantum_mode",
            query="test query",
            documents=sample_corpus["docs"],
            bm25_index=sample_corpus["bm25_idx"],
        )


def test_chunking_granularity():
    """Tests that chunking produces varying chunk counts for small vs baseline vs large sizes."""
    text = (
        "ACME Corp Remote Work Policy. " * 30 +
        "Section 1: Hardware Equipment. All full-time employees receive a $500 stipend. " * 20 +
        "Section 2: Security & Encryption. FileVault or BitLocker encryption is required on all devices. " * 20
    )
    raw_doc = Document(page_content=text, metadata={"source": "long_policy.txt", "page": 1})
    
    chunks_small = chunk_documents([raw_doc], chunk_size=300, chunk_overlap=50)
    chunks_base = chunk_documents([raw_doc], chunk_size=600, chunk_overlap=100)
    chunks_large = chunk_documents([raw_doc], chunk_size=1000, chunk_overlap=150)
    
    assert len(chunks_small) > len(chunks_base) > len(chunks_large)
    for c in chunks_small:
        assert "chunk_id" in c.metadata
        assert len(c.page_content) <= 350
