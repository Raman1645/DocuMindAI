"""
metrics.py - Standalone Information Retrieval (IR) & Generation Metric Suite for RAG.

Implements standard mathematical metrics to evaluate retrieval performance
independently from LLM generation without synthetic/faked numbers:
- Recall@K: Proportion of expected ground-truth facts retrieved in top K.
- Precision@K: Proportion of top K retrieved chunks that are relevant.
- MRR (Mean Reciprocal Rank): Reciprocal rank (1/rank) of first relevant chunk.
- Hit Rate@K: Binary indicator if at least one relevant chunk was found.
- Context Precision: Weighted rank position score of relevant chunks.
"""

from typing import List, Dict, Any, Set, Tuple


def _extract_text_content(chunk_item: Any) -> str:
    """Extracts raw text content from either a Document object or a string/tuple."""
    if hasattr(chunk_item, "page_content"):
        return chunk_item.page_content.lower()
    if isinstance(chunk_item, tuple) and len(chunk_item) > 0:
        first = chunk_item[0]
        if hasattr(first, "page_content"):
            return first.page_content.lower()
        return str(first).lower()
    if isinstance(chunk_item, dict):
        return str(chunk_item.get("content", chunk_item.get("page_content", ""))).lower()
    return str(chunk_item).lower()


def is_chunk_relevant(chunk_content: str, expected_keywords: List[str], match_threshold: float = 0.5) -> bool:
    """
    Determines if a retrieved chunk contains the expected key factual concepts.
    Returns True if at least match_threshold (default 50%) of non-trivial expected keywords match.
    """
    if not expected_keywords:
        return False
    
    cleaned_keywords = [k.lower().strip() for k in expected_keywords if len(k.strip()) > 1]
    if not cleaned_keywords:
        return False
        
    matched = [k for k in cleaned_keywords if k in chunk_content.lower()]
    match_ratio = len(matched) / len(cleaned_keywords)
    return match_ratio >= match_threshold


def compute_hit_rate_at_k(retrieved_chunks: List[Any], expected_keywords: List[str], k: int = 5) -> float:
    """
    Hit Rate@K: Returns 1.0 if at least one chunk in top-K is relevant, else 0.0.
    """
    top_k = retrieved_chunks[:k]
    for chunk in top_k:
        chunk_text = _extract_text_content(chunk)
        if is_chunk_relevant(chunk_text, expected_keywords):
            return 1.0
    return 0.0


def compute_recall_at_k(retrieved_chunks: List[Any], expected_keywords: List[str], k: int = 5) -> float:
    """
    Recall@K: Measures the ratio of unique expected keywords covered across all top-K retrieved chunks.
    """
    if not expected_keywords:
        return 1.0
        
    top_k = retrieved_chunks[:k]
    combined_context = " ".join([_extract_text_content(c) for c in top_k])
    
    cleaned_keywords = [k.lower().strip() for k in expected_keywords if len(k.strip()) > 1]
    if not cleaned_keywords:
        return 1.0
        
    matched_count = sum(1 for kw in cleaned_keywords if kw in combined_context)
    return round(matched_count / len(cleaned_keywords), 4)


def compute_precision_at_k(retrieved_chunks: List[Any], expected_keywords: List[str], k: int = 5) -> float:
    """
    Precision@K: Measures the proportion of the top-K chunks that contain relevant information.
    """
    if not retrieved_chunks or k <= 0:
        return 0.0
        
    top_k = retrieved_chunks[:k]
    actual_k = len(top_k)
    relevant_chunks = 0
    
    for chunk in top_k:
        chunk_text = _extract_text_content(chunk)
        if is_chunk_relevant(chunk_text, expected_keywords):
            relevant_chunks += 1
            
    return round(relevant_chunks / actual_k, 4)


def compute_reciprocal_rank(retrieved_chunks: List[Any], expected_keywords: List[str], k: int = 5) -> float:
    """
    Reciprocal Rank (RR@K): Returns 1 / rank of the first relevant chunk in top-K (1-indexed).
    Returns 0.0 if no relevant chunk is found.
    """
    top_k = retrieved_chunks[:k]
    for rank_idx, chunk in enumerate(top_k, start=1):
        chunk_text = _extract_text_content(chunk)
        if is_chunk_relevant(chunk_text, expected_keywords):
            return round(1.0 / rank_idx, 4)
    return 0.0


def compute_context_precision(retrieved_chunks: List[Any], expected_keywords: List[str], k: int = 5) -> float:
    """
    Context Precision: Rank-weighted average precision of retrieved relevant chunks in top-K.
    Penalizes relevant chunks appearing lower in the rank list.
    """
    top_k = retrieved_chunks[:k]
    if not top_k or not expected_keywords:
        return 0.0
        
    hits = 0
    running_precisions = []
    
    for rank_idx, chunk in enumerate(top_k, start=1):
        chunk_text = _extract_text_content(chunk)
        if is_chunk_relevant(chunk_text, expected_keywords):
            hits += 1
            running_precisions.append(hits / rank_idx)
            
    if not running_precisions:
        return 0.0
        
    return round(sum(running_precisions) / len(top_k), 4)


def evaluate_retrieval_dataset(
    eval_items: List[Dict[str, Any]],
    retrieval_fn: Any,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Evaluates a retrieval pipeline across an entire evaluation dataset.
    
    Args:
        eval_items: List of benchmark question dictionaries from dataset.json.
        retrieval_fn: Callable taking `query: str` and returning list of retrieved chunks.
        k: Top-K cutoff for evaluation metrics.
        
    Returns:
        Dict containing overall aggregate metrics, category breakdowns, and per-query details.
    """
    import time
    
    results = []
    category_metrics: Dict[str, Dict[str, List[float]]] = {}
    
    for item in eval_items:
        qid = item.get("id", "")
        question = item["question"]
        category = item.get("category", "general")
        expected_keywords = item.get("expected_keywords", [])
        is_answerable = item.get("is_answerable", True)
        
        # Measure retrieval latency
        t_start = time.perf_counter()
        retrieved_chunks = retrieval_fn(question)
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        
        # Calculate isolated metrics
        if is_answerable:
            hit = compute_hit_rate_at_k(retrieved_chunks, expected_keywords, k=k)
            recall = compute_recall_at_k(retrieved_chunks, expected_keywords, k=k)
            precision = compute_precision_at_k(retrieved_chunks, expected_keywords, k=k)
            mrr = compute_reciprocal_rank(retrieved_chunks, expected_keywords, k=k)
            ctx_precision = compute_context_precision(retrieved_chunks, expected_keywords, k=k)
        else:
            # For unanswerable queries, expected recall is N/A or abstention-targeted
            hit = 0.0
            recall = 0.0
            precision = 0.0
            mrr = 0.0
            ctx_precision = 0.0
            
        record = {
            "id": qid,
            "category": category,
            "question": question,
            "is_answerable": is_answerable,
            "latency_ms": latency_ms,
            "hit_rate": hit,
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
            "context_precision": ctx_precision,
            "retrieved_count": len(retrieved_chunks),
        }
        results.append(record)
        
        # Group by category for answerable items
        if is_answerable:
            if category not in category_metrics:
                category_metrics[category] = {"hit_rate": [], "recall": [], "precision": [], "mrr": [], "context_precision": [], "latency_ms": []}
            category_metrics[category]["hit_rate"].append(hit)
            category_metrics[category]["recall"].append(recall)
            category_metrics[category]["precision"].append(precision)
            category_metrics[category]["mrr"].append(mrr)
            category_metrics[category]["context_precision"].append(ctx_precision)
            category_metrics[category]["latency_ms"].append(latency_ms)

    # Compute overall macro averages across answerable items
    answerable_records = [r for r in results if r["is_answerable"]]
    total_count = len(answerable_records) if answerable_records else 1
    
    overall_summary = {
        "total_questions": len(results),
        "answerable_questions": len(answerable_records),
        "unanswerable_questions": len(results) - len(answerable_records),
        "mean_hit_rate": round(sum(r["hit_rate"] for r in answerable_records) / total_count, 4),
        "mean_recall_at_k": round(sum(r["recall"] for r in answerable_records) / total_count, 4),
        "mean_precision_at_k": round(sum(r["precision"] for r in answerable_records) / total_count, 4),
        "mean_mrr": round(sum(r["mrr"] for r in answerable_records) / total_count, 4),
        "mean_context_precision": round(sum(r["context_precision"] for r in answerable_records) / total_count, 4),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 2),
    }
    
    # Compute per-category averages
    category_summary = {}
    for cat, metrics in category_metrics.items():
        cat_count = len(metrics["hit_rate"])
        category_summary[cat] = {
            "count": cat_count,
            "hit_rate": round(sum(metrics["hit_rate"]) / cat_count, 4),
            "recall": round(sum(metrics["recall"]) / cat_count, 4),
            "precision": round(sum(metrics["precision"]) / cat_count, 4),
            "mrr": round(sum(metrics["mrr"]) / cat_count, 4),
            "context_precision": round(sum(metrics["context_precision"]) / cat_count, 4),
            "avg_latency_ms": round(sum(metrics["latency_ms"]) / cat_count, 2),
        }
        
    return {
        "overall": overall_summary,
        "by_category": category_summary,
        "detailed_results": results,
    }
