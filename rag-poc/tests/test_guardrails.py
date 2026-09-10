"""
test_guardrails.py - Unit and Integration Tests for Phase 4 Guardrail Suite.

Validates that:
1. Conversational Query Rewriter detects references and reformulates multi-turn questions.
2. Confidence-Aware Abstention Gate detects low-relevance candidates and triggers deterministic abstention.
3. Citation & Evidence Verification Layer flags unsupported claims and validates accurate citations.
4. generate_answer orchestrates guardrails seamlessly.
"""

import pytest
from langchain_core.documents import Document

from query_rewriter import contains_conversational_reference, rewrite_query
from confidence import check_retrieval_confidence
from verifier import verify_grounded_answer
from generation import generate_answer, sanitize_model_output


def test_sanitize_model_output():
    """Tests that sanitize_model_output cleanly strips <think> blocks, prefixes, and handles edge cases."""
    # 1. Closed <think> block
    raw1 = "<think>\nLet me analyze the documents.\n1. Look at policy.\n</think>\nFull-time employees receive $500 [Source 1]."
    assert sanitize_model_output(raw1) == "Full-time employees receive $500 [Source 1]."

    # 2. Case-insensitive <THINK> tags
    raw2 = "<THINK>internal reasoning</THINK>Core hours are 10 AM to 4 PM [Source 2]."
    assert sanitize_model_output(raw2) == "Core hours are 10 AM to 4 PM [Source 2]."

    # 3. Unclosed <think> block (token limit cutoff)
    raw3 = "<think>Generating response and checking citations...\nStill thinking"
    assert sanitize_model_output(raw3) == "I don't know based on the provided documents."

    # 4. Stray closing tag
    raw4 = "Answer text here.</think>"
    assert sanitize_model_output(raw4) == "Answer text here."

    # 5. Thought prefix
    raw5 = "Thought: Here is the plan.\n\nEmployees receive 20 days PTO [Source 1]."
    assert sanitize_model_output(raw5) == "Employees receive 20 days PTO [Source 1]."

    # 6. Standard clean answer
    raw6 = "All full-time employees are eligible for benefits [Source 1]."
    assert sanitize_model_output(raw6) == raw6

    # 7. Standard clean abstention
    raw7 = "I don't know based on the provided documents."
    assert sanitize_model_output(raw7) == raw7


def test_conversational_reference_detection():
    """Tests heuristic detection of conversational pronouns and follow-up triggers."""
    # Contextual follow-up queries
    assert contains_conversational_reference("What about the second one?") is True
    assert contains_conversational_reference("How much do they receive?") is True
    assert contains_conversational_reference("Tell me more about it.") is True
    assert contains_conversational_reference("Which one applies to remote workers?") is True

    # Standalone factual queries
    assert contains_conversational_reference("What is the home office hardware setup stipend?") is False
    assert contains_conversational_reference("How many days of paid time off do full-time employees get?") is False


def test_query_rewriter_mock():
    """Tests query rewriting logic with conversational history."""
    history = [
        {"role": "user", "content": "What is the equipment stipend for remote employees?"},
        {"role": "assistant", "content": "All full-time employees receive a $500 home office equipment stipend."},
    ]
    
    # Standalone query without pronouns should return original query directly
    standalone_q = "What is the 401(k) match?"
    assert rewrite_query(standalone_q, chat_history=history) == standalone_q

    # Query with pronoun should invoke rewrite chain
    follow_up_q = "How many days do I have to submit it?"
    rewritten = rewrite_query(follow_up_q, chat_history=history, provider="mock")
    assert len(rewritten) > 0


def test_confidence_abstention_high_vs_low():
    """Tests that high-score retrieval passes and low-score retrieval triggers abstention."""
    high_chunks = [
        (Document(page_content="Full-time employees receive $500 stipend.", metadata={"chunk_id": "c1"}), 0.85),
    ]
    is_conf, score, msg = check_retrieval_confidence(high_chunks, threshold=-2.5)
    assert is_conf is True
    assert score == 0.85
    assert msg == ""

    # Low score retrieval (out of domain / nonsensical)
    low_chunks = [
        (Document(page_content="Completely unrelated document text.", metadata={"chunk_id": "c2"}), -4.20),
    ]
    is_conf_low, score_low, msg_low = check_retrieval_confidence(low_chunks, threshold=-2.5)
    assert is_conf_low is False
    assert score_low == -4.20
    assert "don't know" in msg_low.lower()


def test_verifier_grounded_vs_unsupported():
    """Tests the Answer Verifier layer."""
    chunks = [
        (Document(page_content="ACME Corp full-time employees receive a $500 home office stipend.", metadata={"chunk_id": "c1"}), 0.9),
        (Document(page_content="VPN access is mandatory for remote workers.", metadata={"chunk_id": "c2"}), 0.8),
    ]

    # Case A: Properly cited and grounded claim
    good_ans = "Full-time employees receive a $500 home office stipend [Source 1]."
    report_good = verify_grounded_answer(good_ans, chunks)
    assert report_good["is_verified"] is True
    assert report_good["valid_citations"] >= 1
    assert len(report_good["unsupported_claims"]) == 0

    # Case B: Claim without citation
    bad_ans_no_cite = "All employees get free sports tickets every Friday."
    report_bad = verify_grounded_answer(bad_ans_no_cite, chunks)
    assert report_bad["is_verified"] is False
    assert len(report_bad["unsupported_claims"]) > 0

    # Case C: Standard abstention answer
    abstain_ans = "I don't know based on the provided documents."
    report_abstain = verify_grounded_answer(abstain_ans, chunks)
    assert report_abstain["is_verified"] is True
    assert report_abstain["status"] == "abstained"


def test_generate_answer_confidence_gate_integration():
    """Tests that generate_answer halts and abstains immediately when confidence is low."""
    low_chunks = [
        (Document(page_content="Unrelated text snippet", metadata={"source": "unrelated.txt", "page": 1}), -6.5),
    ]

    res = generate_answer(
        query="What is the capital of Mars?",
        retrieved_chunks=low_chunks,
        provider="mock",
    )

    assert res["is_confident"] is False
    assert "don't know" in res["answer"].lower()
    assert len(res["sources"]) == 0
    assert res["verification"]["status"] == "abstained_low_confidence"
