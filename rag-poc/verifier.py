"""
verifier.py - Citation & Evidence Verification Layer.

Performs post-generation verification to ensure that every factual claim in the generated answer
is grounded in and explicitly cited to a valid, relevant source context chunk.
"""

import re
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

from evaluation.metrics import _extract_text_content


def verify_grounded_answer(
    answer_text: str,
    retrieved_chunks: List[Tuple[Document, float]],
) -> Dict[str, Any]:
    """
    Verifies that claims in the answer are cited and textually supported by referenced chunks.

    Args:
        answer_text: Raw answer string from LLM.
        retrieved_chunks: Top-N reranked (Document, score) candidate chunks.

    Returns:
        Dict[str, Any]:
            - is_verified: bool (True if all citations are valid and supported)
            - total_citations: int
            - valid_citations: int
            - cited_sources: list of source tags
            - unsupported_claims: list of sentences with missing or invalid citations
    """
    if not answer_text or not retrieved_chunks:
        return {
            "is_verified": False,
            "total_citations": 0,
            "valid_citations": 0,
            "cited_sources": [],
            "unsupported_claims": [],
        }

    # Check for standard abstention
    if "i don't know" in answer_text.lower() or "not provided" in answer_text.lower():
        return {
            "is_verified": True,
            "total_citations": 0,
            "valid_citations": 0,
            "cited_sources": [],
            "unsupported_claims": [],
            "status": "abstained",
        }

    # Split answer into sentence-level claims
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if len(s.strip()) > 10]
    
    total_citations = 0
    valid_citations = 0
    all_cited_tags = []
    unsupported_claims = []

    for sentence in sentences:
        cited_indices = [int(m) for m in re.findall(r"\[Source\s+(\d+)\]", sentence, re.IGNORECASE)]
        
        if not cited_indices:
            # Sentence makes a claim without citation
            unsupported_claims.append({
                "sentence": sentence,
                "reason": "Missing citation",
            })
            continue

        sentence_valid = True
        for idx in cited_indices:
            total_citations += 1
            all_cited_tags.append(f"[Source {idx}]")
            chunk_idx = idx - 1
            
            if 0 <= chunk_idx < len(retrieved_chunks):
                doc, _score = retrieved_chunks[chunk_idx]
                chunk_text = _extract_text_content(doc)
                
                # Extract key nouns/numbers from sentence to check presence in chunk
                sentence_keywords = [
                    w for w in re.findall(r"\w+", sentence.lower())
                    if len(w) > 3 and w not in {"based", "provided", "documents", "source", "according"}
                ]
                
                if sentence_keywords:
                    overlap = sum(1 for kw in sentence_keywords if kw in chunk_text)
                    if overlap / len(sentence_keywords) >= 0.25:
                        valid_citations += 1
                    else:
                        sentence_valid = False
            else:
                sentence_valid = False

        if not sentence_valid:
            unsupported_claims.append({
                "sentence": sentence,
                "reason": "Cited source index out of bounds or irrelevant",
            })

    is_verified = (len(unsupported_claims) == 0) and (total_citations > 0)
    
    return {
        "is_verified": is_verified,
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "citation_precision": round(valid_citations / total_citations, 4) if total_citations > 0 else 0.0,
        "cited_sources": sorted(list(set(all_cited_tags))),
        "unsupported_claims": unsupported_claims,
    }
