"""
test_metrics.py - Unit tests for Phase 1 IR evaluation metrics.

Verifies Hit Rate@K, Recall@K, Precision@K, MRR, and Context Precision calculations
under deterministic dummy retrieval conditions.
"""

import sys
from pathlib import Path

# Ensure rag-poc root is on sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import pytest
from evaluation.metrics import (
    compute_hit_rate_at_k,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_reciprocal_rank,
    compute_context_precision,
    is_chunk_relevant,
)


def test_chunk_relevance():
    chunk_text = "All full-time employees receive a one-time stipend of $500 for ergonomic home office gear."
    keywords = ["$500", "stipend", "ergonomic", "gear"]
    assert is_chunk_relevant(chunk_text, keywords, match_threshold=0.5) is True
    
    unrelated_keywords = ["pet insurance", "dog", "cat"]
    assert is_chunk_relevant(chunk_text, unrelated_keywords, match_threshold=0.5) is False


def test_hit_rate_at_k():
    chunks = [
        "Unrelated document about general meeting times.",
        "The $500 home office equipment stipend is available for full-time employees.",
        "Another unrelated chunk on parking.",
    ]
    keywords = ["$500", "stipend", "equipment"]
    
    assert compute_hit_rate_at_k(chunks, keywords, k=1) == 0.0
    assert compute_hit_rate_at_k(chunks, keywords, k=2) == 1.0
    assert compute_hit_rate_at_k(chunks, keywords, k=3) == 1.0


def test_recall_at_k():
    chunks = [
        "ACME offers $500 stipend for ergonomic desks.",
        "Full-time workers can purchase monitors and chairs.",
    ]
    keywords = ["$500", "stipend", "desks", "monitors", "chairs"]
    
    # 3 keywords in chunk 1, 2 in chunk 2
    recall_k1 = compute_recall_at_k(chunks, keywords, k=1)
    assert recall_k1 == 0.6  # 3/5 = 0.6
    
    recall_k2 = compute_recall_at_k(chunks, keywords, k=2)
    assert recall_k2 == 1.0  # 5/5 = 1.0


def test_precision_at_k():
    chunks = [
        "The $500 equipment stipend for desks and chairs.",  # Relevant
        "General company introduction and greeting.",      # Irrelevant
        "Core hours are 10 AM to 4 PM EST.",               # Irrelevant
        "All $500 hardware purchases must be logged.",      # Relevant
    ]
    keywords = ["$500", "stipend", "hardware", "desks"]
    
    # K=1: 1/1 = 1.0
    assert compute_precision_at_k(chunks, keywords, k=1) == 1.0
    # K=2: 1/2 = 0.5
    assert compute_precision_at_k(chunks, keywords, k=2) == 0.5
    # K=4: 2/4 = 0.5
    assert compute_precision_at_k(chunks, keywords, k=4) == 0.5


def test_mrr():
    chunks_first = [
        "Relevant chunk containing $500 stipend ergonomic.",
        "Irrelevant chunk.",
    ]
    chunks_second = [
        "Irrelevant chunk.",
        "Relevant chunk containing $500 stipend ergonomic.",
    ]
    keywords = ["$500", "stipend", "ergonomic"]
    
    assert compute_reciprocal_rank(chunks_first, keywords, k=3) == 1.0
    assert compute_reciprocal_rank(chunks_second, keywords, k=3) == 0.5
    assert compute_reciprocal_rank(["irrelevant only"], keywords, k=3) == 0.0


def test_context_precision():
    # Relevant chunk at rank 1: Precision@1 = 1.0 / 1 = 1.0
    chunks = [
        "Relevant chunk containing $500 stipend ergonomic.",
        "Irrelevant chunk.",
    ]
    keywords = ["$500", "stipend", "ergonomic"]
    ctx_prec = compute_context_precision(chunks, keywords, k=2)
    assert ctx_prec == 0.5  # (1.0) / 2 = 0.5
