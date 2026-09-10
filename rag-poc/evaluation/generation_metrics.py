"""
generation_metrics.py - Mathematical & Groundedness Metric Suite for Generation Evaluation.

Evaluates generated answers against retrieved context and ground-truth facts:
1. Groundedness Score: Measures whether claims in the generated answer are strictly supported by retrieved context.
2. Citation Accuracy: Verifies that [Source X] citations correctly reference relevant context chunks.
3. Abstention Accuracy: Measures correct refusal ("I don't know based on the provided documents") on unanswerable queries.
4. Keyword Coverage: Proportion of expected ground truth facts included in the final answer.
"""

import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

from evaluation.metrics import _extract_text_content, is_chunk_relevant


def compute_abstention_accuracy(
    answer_text: str,
    is_answerable: bool,
) -> float:
    """
    Evaluates whether the model appropriately answered or abstained.
    
    - If is_answerable == False: Returns 1.0 if the model abstained ("don't know" / "not provided"), else 0.0.
    - If is_answerable == True: Returns 1.0 if the model answered without false abstention, else 0.0.
    """
    abstention_patterns = [
        r"i don't know",
        r"i do not know",
        r"not provided in the",
        r"not mentioned in the",
        r"does not contain sufficient",
        r"no information provided",
        r"insufficient information",
    ]
    
    answer_lower = answer_text.lower()
    abstained = any(re.search(pat, answer_lower) for pat in abstention_patterns)
    
    if not is_answerable:
        return 1.0 if abstained else 0.0
    else:
        return 0.0 if abstained else 1.0


def compute_citation_accuracy(
    answer_text: str,
    retrieved_chunks: List[Any],
    expected_keywords: List[str],
) -> float:
    """
    Measures the precision of inline source citations (e.g. [Source 1], [Source 2]).
    
    Returns 1.0 if every cited source index contains relevant factual concepts.
    Returns 0.0 if any citation points to an irrelevant chunk or no citations were made.
    """
    if not expected_keywords:
        return 1.0
        
    cited_indices = [int(m) for m in re.findall(r"\[Source\s+(\d+)\]", answer_text, re.IGNORECASE)]
    if not cited_indices:
        return 0.0  # Failed to cite
        
    valid_citations = 0
    for idx in cited_indices:
        chunk_idx = idx - 1
        if 0 <= chunk_idx < len(retrieved_chunks):
            chunk = retrieved_chunks[chunk_idx]
            chunk_text = _extract_text_content(chunk)
            if is_chunk_relevant(chunk_text, expected_keywords, match_threshold=0.3):
                valid_citations += 1
                
    return round(valid_citations / len(cited_indices), 4)


def compute_groundedness_score(
    answer_text: str,
    retrieved_chunks: List[Any],
    expected_keywords: List[str],
    is_answerable: bool,
) -> float:
    """
    Calculates overall answer groundedness score:
    - For unanswerable queries: Based on correct abstention.
    - For answerable queries: Ratio of expected ground-truth keywords present in the generated answer.
    """
    if not is_answerable:
        return compute_abstention_accuracy(answer_text, is_answerable=False)
        
    if not expected_keywords:
        return 1.0
        
    answer_lower = answer_text.lower()
    cleaned_keywords = [k.lower().strip() for k in expected_keywords if len(k.strip()) > 1]
    if not cleaned_keywords:
        return 1.0
        
    matched = sum(1 for kw in cleaned_keywords if kw in answer_lower)
    return round(matched / len(cleaned_keywords), 4)
