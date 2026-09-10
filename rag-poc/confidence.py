"""
confidence.py - Confidence-Aware Retrieval Gate & Deterministic Abstention.

Inspects cross-encoder relevance logits from retrieval. If candidate documents
fall below the relevance threshold (indicating out-of-domain / unanswerable questions),
the system deterministically triggers abstention before calling the LLM generator.
"""

from typing import List, Tuple, Dict, Any, Optional
from langchain_core.documents import Document

from config import CONFIDENCE_THRESHOLD

DEFAULT_ABSTENTION_MESSAGE = "I don't know based on the provided documents."


def check_retrieval_confidence(
    retrieved_chunks: List[Tuple[Document, float]],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[bool, float, str]:
    """
    Evaluates cross-encoder relevance scores to determine if context is sufficient.

    Args:
        retrieved_chunks: Top-N reranked (Document, cross_encoder_score) pairs.
        threshold: Minimum acceptable cross-encoder logit for confident generation.

    Returns:
        Tuple[bool, float, str]:
            - is_confident: True if top chunk meets or exceeds threshold, else False.
            - max_score: Highest cross-encoder score among candidates.
            - abstention_msg: Standard abstention message if not confident, else "".
    """
    if not retrieved_chunks:
        return False, -999.0, DEFAULT_ABSTENTION_MESSAGE

    scores = [score for _doc, score in retrieved_chunks]
    max_score = max(scores)

    if max_score < threshold:
        return False, max_score, DEFAULT_ABSTENTION_MESSAGE

    return True, max_score, ""
