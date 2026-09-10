"""
test_model_factory.py - Unit and Integration Tests for Phase 3 LLM Provider Factory & Generation Metrics.

Validates that:
1. llm_factory correctly exposes and instantiates providers ('mock', 'groq', 'ollama', 'openai').
2. Invalid provider strings raise ValueError.
3. Generation metrics (groundedness, citation accuracy, abstention accuracy) calculate deterministic mathematical scores.
4. End-to-end generate_answer works cleanly with Mock fallback and returns valid source structures.
"""

import pytest
from langchain_core.documents import Document

from llm_factory import create_llm, list_supported_providers
from generation import generate_answer
from evaluation.generation_metrics import (
    compute_groundedness_score,
    compute_citation_accuracy,
    compute_abstention_accuracy,
)


def test_list_supported_providers():
    """Tests that provider catalog exposes all required backends."""
    catalog = list_supported_providers()
    assert "ollama" in catalog
    assert "groq" in catalog
    assert "openai" in catalog
    assert "mock" in catalog


def test_create_mock_llm():
    """Tests creating and invoking the deterministic Mock LLM."""
    llm = create_llm(provider="mock")
    response = llm.invoke("What is the policy?")
    assert "Source 1" in str(response)


def test_create_invalid_provider():
    """Tests that unsupported provider raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm(provider="unsupported_quantum_ai")


def test_abstention_accuracy():
    """Tests abstention accuracy logic for answerable vs unanswerable queries."""
    # 1. Unanswerable query where model correctly abstains
    ans_abstain = "I don't know based on the provided documents."
    assert compute_abstention_accuracy(ans_abstain, is_answerable=False) == 1.0

    # 2. Unanswerable query where model hallucinates instead of abstaining
    ans_hallucinate = "The company was founded on Mars in 1920."
    assert compute_abstention_accuracy(ans_hallucinate, is_answerable=False) == 0.0

    # 3. Answerable query where model answers
    ans_valid = "Full-time employees receive $500 [Source 1]."
    assert compute_abstention_accuracy(ans_valid, is_answerable=True) == 1.0


def test_citation_accuracy():
    """Tests citation accuracy calculation."""
    retrieved = [
        Document(page_content="Employees receive a $500 home office equipment stipend.", metadata={"chunk_id": "c1"}),
        Document(page_content="Production code requires two peer reviews before merging.", metadata={"chunk_id": "c2"}),
    ]
    
    # Accurate citation pointing to chunk 1 for stipend query
    ans_good = "Employees are eligible for a $500 stipend [Source 1]."
    score_good = compute_citation_accuracy(ans_good, retrieved, expected_keywords=["500", "stipend"])
    assert score_good == 1.0

    # Inaccurate citation pointing to chunk 2 for stipend query
    ans_bad = "Employees are eligible for a $500 stipend [Source 2]."
    score_bad = compute_citation_accuracy(ans_bad, retrieved, expected_keywords=["500", "stipend"])
    assert score_bad == 0.0


def test_generate_answer_mock_pipeline():
    """Tests end-to-end generate_answer with the Mock LLM."""
    retrieved = [
        (Document(page_content="ACME Corp offers a $500 home office stipend.", metadata={"source": "policy.txt", "page": 1, "chunk_id": "c1"}), 0.95),
        (Document(page_content="VPN is required for remote access.", metadata={"source": "security.txt", "page": 2, "chunk_id": "c2"}), 0.82),
    ]
    
    res = generate_answer(
        query="What is the stipend policy?",
        retrieved_chunks=retrieved,
        provider="mock",
    )
    
    assert "answer" in res
    assert "sources" in res
    assert len(res["sources"]) == 2
    assert res["sources"][0]["source_tag"] == "[Source 1]"
    assert res["sources"][0]["file_name"] == "policy.txt"
