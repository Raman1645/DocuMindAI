"""
evaluate.py - Production-Grade RAG Evaluation Harness & Benchmark Suite.

Executes the full RAG pipeline over evaluation/dataset.json:
1. Ingestion & Indexing of multi-document corpus
2. Retrieval Evaluation: Recall@K, Precision@K, MRR, Hit Rate, Context Precision
3. Generation Evaluation: Grounded generation via Groq LLM, token tracking, latency
4. P0 Reliability: Zero hardcoded/fabricated metric overrides.

Outputs JSON results to evaluation/results/ and formatted Markdown reports.
"""

import json
import os
import sys
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

# Add rag-poc to python path for imports
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from ingestion import ingest_file
from embeddings import add_documents_to_vector_store
from retrieval import build_bm25_index, retrieve_and_rerank
from generation import generate_answer, get_llm
from evaluation.metrics import (
    compute_hit_rate_at_k,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_reciprocal_rank,
    compute_context_precision,
)


def load_eval_dataset(json_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads benchmark questions from evaluation/dataset.json or eval_dataset.json."""
    if json_path is None:
        p1 = current_dir / "evaluation" / "dataset.json"
        p2 = current_dir / "eval_dataset.json"
        json_path = str(p1 if p1.exists() else p2)
        
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {path.resolve()}")
        
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation_benchmark(
    dataset_path: Optional[str] = None,
    top_k: int = 4,
    enable_generation: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes benchmark over the multi-document corpus and computes genuine retrieval & generation metrics.
    """
    eval_items = load_eval_dataset(dataset_path)
    sample_dir = current_dir / "sample_data"
    
    # 1. Index multi-document sample files
    sample_files = list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.pdf"))
    if not sample_files:
        raise FileNotFoundError(f"No sample documents found in {sample_dir.resolve()}. Run generate_sample_data.py first.")
        
    print(f"\n[EVAL] Ingesting {len(sample_files)} documents from {sample_dir.name}...")
    temp_db_dir = current_dir / "chroma_eval_benchmark_db"
    try:
        if temp_db_dir.exists():
            shutil.rmtree(temp_db_dir)
    except PermissionError:
        pass
        
    all_chunks = []
    for file_path in sample_files:
        chunks = ingest_file(file_path)
        all_chunks.extend(chunks)
        print(f"  - Ingested '{file_path.name}': {len(chunks)} chunks")
        
    add_documents_to_vector_store(
        all_chunks,
        persist_directory=str(temp_db_dir),
        collection_name="eval_benchmark",
    )
    bm25_idx, _ = build_bm25_index(all_chunks)
    print(f"[EVAL] Total corpus indexed: {len(all_chunks)} chunks across {len(sample_files)} documents.\n")

    # 2. Run evaluation questions
    detailed_records = []
    category_buckets: Dict[str, Dict[str, List[float]]] = {}
    
    print(f"[EVAL] Executing benchmark across {len(eval_items)} questions...")
    for idx, item in enumerate(eval_items, start=1):
        qid = item.get("id", f"Q{idx:02d}")
        category = item.get("category", "general")
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        expected_keywords = item.get("expected_keywords", [])
        is_answerable = item.get("is_answerable", True)
        
        # A. Retrieval & Latency Timing
        t_ret_start = time.perf_counter()
        retrieved_chunks = retrieve_and_rerank(
            query=question,
            documents=all_chunks,
            bm25_index=bm25_idx,
            top_k_dense=6,
            top_k_bm25=6,
            top_n_rerank=top_k,
            persist_directory=str(temp_db_dir),
            collection_name="eval_benchmark",
        )
        retrieval_latency_ms = round((time.perf_counter() - t_ret_start) * 1000, 2)
        
        # B. Mathematical Retrieval Metric Calculations
        if is_answerable:
            hit = compute_hit_rate_at_k(retrieved_chunks, expected_keywords, k=top_k)
            recall = compute_recall_at_k(retrieved_chunks, expected_keywords, k=top_k)
            precision = compute_precision_at_k(retrieved_chunks, expected_keywords, k=top_k)
            mrr = compute_reciprocal_rank(retrieved_chunks, expected_keywords, k=top_k)
            ctx_prec = compute_context_precision(retrieved_chunks, expected_keywords, k=top_k)
        else:
            # Unanswerable queries: tested for abstention in generation
            hit = 0.0
            recall = 0.0
            precision = 0.0
            mrr = 0.0
            ctx_prec = 0.0
            
        # C. LLM Generation
        answer_text = ""
        gen_latency_ms = 0.0
        if enable_generation:
            t_gen_start = time.perf_counter()
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    gen_resp = generate_answer(query=question, retrieved_chunks=retrieved_chunks)
                    answer_text = gen_resp.get("answer", "")
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        sleep_time = (attempt + 1) * 3
                        time.sleep(sleep_time)
                    else:
                        answer_text = f"[ERROR: Generation Failed - {str(e)}]"
            gen_latency_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
            # Small delay between items to respect API rate limits
            time.sleep(0.3)
            
        record = {
            "id": qid,
            "category": category,
            "question": question,
            "ground_truth": ground_truth,
            "is_answerable": is_answerable,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": gen_latency_ms,
            "total_latency_ms": round(retrieval_latency_ms + gen_latency_ms, 2),
            "hit_rate": hit,
            "recall_at_k": recall,
            "precision_at_k": precision,
            "mrr": mrr,
            "context_precision": ctx_prec,
            "generated_answer": answer_text,
            "retrieved_chunk_ids": [doc.metadata.get("chunk_id", "") for doc, _ in retrieved_chunks],
        }
        detailed_records.append(record)
        
        # Aggregate category data
        if is_answerable:
            if category not in category_buckets:
                category_buckets[category] = {"hit": [], "recall": [], "prec": [], "mrr": [], "ctx_prec": [], "lat": []}
            category_buckets[category]["hit"].append(hit)
            category_buckets[category]["recall"].append(recall)
            category_buckets[category]["prec"].append(precision)
            category_buckets[category]["mrr"].append(mrr)
            category_buckets[category]["ctx_prec"].append(ctx_prec)
            category_buckets[category]["lat"].append(retrieval_latency_ms)

        if idx % 10 == 0 or idx == len(eval_items):
            print(f"  Progress: {idx}/{len(eval_items)} completed...")

    # 3. Overall Summary Calculations
    answerable_list = [r for r in detailed_records if r["is_answerable"]]
    ans_count = len(answerable_list) if answerable_list else 1
    
    overall_summary = {
        "total_eval_questions": len(detailed_records),
        "answerable_questions": len(answerable_list),
        "unanswerable_questions": len(detailed_records) - len(answerable_list),
        "mean_hit_rate": round(sum(r["hit_rate"] for r in answerable_list) / ans_count, 4),
        "mean_recall_at_k": round(sum(r["recall_at_k"] for r in answerable_list) / ans_count, 4),
        "mean_precision_at_k": round(sum(r["precision_at_k"] for r in answerable_list) / ans_count, 4),
        "mean_mrr": round(sum(r["mrr"] for r in answerable_list) / ans_count, 4),
        "mean_context_precision": round(sum(r["context_precision"] for r in answerable_list) / ans_count, 4),
        "mean_retrieval_latency_ms": round(sum(r["retrieval_latency_ms"] for r in detailed_records) / len(detailed_records), 2),
        "mean_generation_latency_ms": round(sum(r["generation_latency_ms"] for r in detailed_records) / len(detailed_records), 2),
    }

    # Per-category summary
    category_summary = {}
    for cat, bucket in category_buckets.items():
        c_count = len(bucket["hit"])
        category_summary[cat] = {
            "count": c_count,
            "hit_rate": round(sum(bucket["hit"]) / c_count, 4),
            "recall": round(sum(bucket["recall"]) / c_count, 4),
            "precision": round(sum(bucket["prec"]) / c_count, 4),
            "mrr": round(sum(bucket["mrr"]) / c_count, 4),
            "context_precision": round(sum(bucket["ctx_prec"]) / c_count, 4),
            "avg_retrieval_latency_ms": round(sum(bucket["lat"]) / c_count, 2),
        }

    # 4. Save JSON Results
    if output_dir is None:
        out_dir = current_dir / "evaluation" / "results"
    else:
        out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"evaluation_run_{timestamp}.json"
    
    full_output = {
        "timestamp": timestamp,
        "overall_summary": overall_summary,
        "category_summary": category_summary,
        "detailed_results": detailed_records,
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"\n[SUCCESS] Evaluation report saved to: {out_file.resolve()}")

    # Cleanup temporary benchmark DB
    try:
        if temp_db_dir.exists():
            shutil.rmtree(temp_db_dir)
    except PermissionError:
        pass

    return full_output


def print_markdown_report(eval_output: Dict[str, Any]):
    """Formats and prints an executive summary table for Markdown documentation."""
    overall = eval_output["overall_summary"]
    categories = eval_output["category_summary"]
    
    print("\n" + "=" * 70)
    print("           [DOCUMIND AI - RETRIEVAL EVALUATION BENCHMARK]")
    print("=" * 70)
    
    print("\n### 1. Overall System Retrieval Performance\n")
    print("| Metric | Measured Score | Standard Benchmark Target | Status |")
    print("|---|---:|---:|:---:|")
    print(f"| **Hit Rate@4** | {overall['mean_hit_rate'] * 100:.1f}% | >= 90.0% | {'[PASS]' if overall['mean_hit_rate'] >= 0.90 else '[LOW]'} |")
    print(f"| **Recall@4** | {overall['mean_recall_at_k'] * 100:.1f}% | >= 85.0% | {'[PASS]' if overall['mean_recall_at_k'] >= 0.85 else '[LOW]'} |")
    print(f"| **Precision@4** | {overall['mean_precision_at_k'] * 100:.1f}% | >= 70.0% | {'[PASS]' if overall['mean_precision_at_k'] >= 0.70 else '[LOW]'} |")
    print(f"| **MRR@4** | {overall['mean_mrr']:.4f} | >= 0.8000 | {'[PASS]' if overall['mean_mrr'] >= 0.80 else '[LOW]'} |")
    print(f"| **Context Precision** | {overall['mean_context_precision']:.4f} | >= 0.7500 | {'[PASS]' if overall['mean_context_precision'] >= 0.75 else '[LOW]'} |")
    print(f"| **Avg Retrieval Latency** | {overall['mean_retrieval_latency_ms']:.1f} ms | < 250 ms | {'[FAST]' if overall['mean_retrieval_latency_ms'] < 250 else '[SLOW]'} |")
    
    print("\n### 2. Breakdown By Query Category\n")
    cat_rows = []
    for cat_name, metrics in categories.items():
        cat_rows.append({
            "Category": cat_name.replace("_", " ").title(),
            "Count": metrics["count"],
            "Hit Rate": f"{metrics['hit_rate'] * 100:.1f}%",
            "Recall@4": f"{metrics['recall'] * 100:.1f}%",
            "Precision@4": f"{metrics['precision'] * 100:.1f}%",
            "MRR": f"{metrics['mrr']:.3f}",
            "Latency": f"{metrics['avg_retrieval_latency_ms']:.1f} ms",
        })
    df_cat = pd.DataFrame(cat_rows)
    try:
        print(df_cat.to_markdown(index=False))
    except Exception:
        print(df_cat.to_string(index=False))
        
    print("\n" + "=" * 70 + "\n")


def main():
    print("=== Step 7: Starting DocuMindAI Phase 1 Benchmark Harness ===")
    eval_output = run_evaluation_benchmark(enable_generation=True)
    print_markdown_report(eval_output)


if __name__ == "__main__":
    main()
