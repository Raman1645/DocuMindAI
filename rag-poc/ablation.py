"""
ablation.py - Retrieval Pipeline & Chunking Strategy Ablation Study Harness.

Executes controlled empirical ablation experiments across the 80-question benchmark:
1. Pipeline Ablation:
   - Config A: Dense-only (ChromaDB + BAAI/bge-small-en-v1.5)
   - Config B: BM25-only (BM25Okapi sparse lexical search)
   - Config C: Hybrid + RRF (Dense + BM25 combined via Reciprocal Rank Fusion)
   - Config D: Hybrid + RRF + Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)

2. Chunking Strategy Ablation:
   - Small: chunk_size=300, chunk_overlap=50
   - Baseline: chunk_size=600, chunk_overlap=100
   - Large: chunk_size=1000, chunk_overlap=150

Outputs formatted Markdown tables and saves persistent JSON results to evaluation/results/.
"""

import json
import os
import sys
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

# Add rag-poc to python path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from ingestion import load_document, chunk_documents
from embeddings import add_documents_to_vector_store
from retrieval import build_bm25_index, retrieve_by_mode
from evaluation.metrics import (
    compute_hit_rate_at_k,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_reciprocal_rank,
    compute_context_precision,
)


def load_benchmark_dataset(dataset_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads benchmark evaluation questions."""
    if dataset_path is None:
        p1 = current_dir / "evaluation" / "dataset.json"
        p2 = current_dir / "eval_dataset.json"
        dataset_path = str(p1 if p1.exists() else p2)
        
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at: {path.resolve()}")
        
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_corpus(
    sample_files: List[Path],
    chunk_size: int = 600,
    chunk_overlap: int = 100,
    db_dir: Path = None,
    collection_name: str = "ablation_test",
) -> Tuple[List[Any], Any, Path]:
    """Chunks documents and indexes them in ChromaDB and BM25Okapi."""
    if db_dir is None:
        db_dir = current_dir / f"chroma_ablation_{collection_name}"
        
    try:
        if db_dir.exists():
            shutil.rmtree(db_dir)
    except PermissionError:
        pass

    all_chunks = []
    for file_path in sample_files:
        raw_docs = load_document(file_path)
        chunks = chunk_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)

    add_documents_to_vector_store(
        all_chunks,
        persist_directory=str(db_dir),
        collection_name=collection_name,
    )
    bm25_idx, _ = build_bm25_index(all_chunks)
    return all_chunks, bm25_idx, db_dir


def evaluate_single_configuration(
    mode: str,
    eval_items: List[Dict[str, Any]],
    documents: List[Any],
    bm25_index: Any,
    persist_dir: str,
    collection_name: str,
    k: int = 4,
) -> Dict[str, Any]:
    """Evaluates a single retrieval configuration across the entire benchmark dataset."""
    detailed_records = []
    latencies = []
    
    # Pre-warm HuggingFace models to avoid counting cold-start initialization in latency
    _ = retrieve_by_mode(
        mode=mode,
        query="warmup test query",
        documents=documents,
        bm25_index=bm25_index,
        top_k=k,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    for item in eval_items:
        qid = item.get("id", "")
        category = item.get("category", "general")
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])
        is_answerable = item.get("is_answerable", True)

        t_start = time.perf_counter()
        retrieved_chunks = retrieve_by_mode(
            mode=mode,
            query=question,
            documents=documents,
            bm25_index=bm25_index,
            top_k=k,
            persist_directory=persist_dir,
            collection_name=collection_name,
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        latencies.append(latency_ms)

        if is_answerable:
            hit = compute_hit_rate_at_k(retrieved_chunks, expected_keywords, k=k)
            recall = compute_recall_at_k(retrieved_chunks, expected_keywords, k=k)
            precision = compute_precision_at_k(retrieved_chunks, expected_keywords, k=k)
            mrr = compute_reciprocal_rank(retrieved_chunks, expected_keywords, k=k)
            ctx_prec = compute_context_precision(retrieved_chunks, expected_keywords, k=k)
        else:
            hit = 0.0
            recall = 0.0
            precision = 0.0
            mrr = 0.0
            ctx_prec = 0.0

        detailed_records.append({
            "id": qid,
            "category": category,
            "is_answerable": is_answerable,
            "latency_ms": latency_ms,
            "hit_rate": hit,
            "recall": recall,
            "precision": precision,
            "mrr": mrr,
            "context_precision": ctx_prec,
            "retrieved_chunk_ids": [doc.metadata.get("chunk_id", "") for doc, _ in retrieved_chunks],
        })

    answerable = [r for r in detailed_records if r["is_answerable"]]
    ans_count = len(answerable) if answerable else 1

    summary = {
        "mode": mode,
        "total_queries": len(detailed_records),
        "hit_rate": round(sum(r["hit_rate"] for r in answerable) / ans_count, 4),
        "recall_at_k": round(sum(r["recall"] for r in answerable) / ans_count, 4),
        "precision_at_k": round(sum(r["precision"] for r in answerable) / ans_count, 4),
        "mrr": round(sum(r["mrr"] for r in answerable) / ans_count, 4),
        "context_precision": round(sum(r["context_precision"] for r in answerable) / ans_count, 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
    }

    return {
        "summary": summary,
        "detailed_records": detailed_records,
    }


def run_pipeline_ablation(
    sample_files: List[Path],
    eval_items: List[Dict[str, Any]],
    top_k: int = 4,
) -> Dict[str, Any]:
    """Runs 4-way retrieval pipeline ablation study."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 1: RETRIEVAL PIPELINE ABLATION STUDY")
    print("=" * 65)
    
    # 1. Prepare baseline corpus (chunk_size=600, chunk_overlap=100)
    print(f"\n[ABLATION] Ingesting & indexing baseline corpus ({len(sample_files)} files)...")
    chunks, bm25_idx, temp_db = prepare_corpus(
        sample_files,
        chunk_size=600,
        chunk_overlap=100,
        collection_name="pipe_ablation",
    )
    print(f"[ABLATION] Corpus ready: {len(chunks)} chunks indexed.\n")

    configurations = [
        ("Dense-Only (BGE-small)", "dense"),
        ("BM25-Only (Sparse)", "bm25"),
        ("Hybrid (Dense + BM25 + RRF)", "hybrid"),
        ("Hybrid + RRF + Cross-Encoder Reranker", "hybrid_rerank"),
    ]

    pipeline_results = {}
    for name, mode in configurations:
        print(f"  --> Benchmarking Config: '{name}' ...", end="", flush=True)
        t0 = time.time()
        res = evaluate_single_configuration(
            mode=mode,
            eval_items=eval_items,
            documents=chunks,
            bm25_index=bm25_idx,
            persist_dir=str(temp_db),
            collection_name="pipe_ablation",
            k=top_k,
        )
        elapsed = round(time.time() - t0, 2)
        print(f" Done ({elapsed}s) | Recall@{top_k}: {res['summary']['recall_at_k']*100:.1f}% | MRR: {res['summary']['mrr']:.4f}")
        pipeline_results[name] = res

    # Cleanup temp db
    try:
        if temp_db.exists():
            shutil.rmtree(temp_db)
    except PermissionError:
        pass

    return pipeline_results


def run_chunking_ablation(
    sample_files: List[Path],
    eval_items: List[Dict[str, Any]],
    top_k: int = 4,
) -> Dict[str, Any]:
    """Runs chunking strategy ablation study across 3 granularity configurations."""
    print("\n" + "=" * 65)
    print("  EXPERIMENT 2: CHUNKING STRATEGY ABLATION STUDY")
    print("=" * 65)

    chunking_configs = [
        ("Small (300 chars, overlap 50)", 300, 50),
        ("Baseline (600 chars, overlap 100)", 600, 100),
        ("Large (1000 chars, overlap 150)", 1000, 150),
    ]

    chunking_results = {}
    for name, size, overlap in chunking_configs:
        print(f"\n[CHUNKING] Indexing corpus with {name}...", flush=True)
        col_name = f"chunk_{size}_{overlap}"
        chunks, bm25_idx, temp_db = prepare_corpus(
            sample_files,
            chunk_size=size,
            chunk_overlap=overlap,
            collection_name=col_name,
        )
        print(f"[CHUNKING] Indexed {len(chunks)} chunks. Running full pipeline benchmark...", end="", flush=True)

        res = evaluate_single_configuration(
            mode="hybrid_rerank",
            eval_items=eval_items,
            documents=chunks,
            bm25_index=bm25_idx,
            persist_dir=str(temp_db),
            collection_name=col_name,
            k=top_k,
        )
        res["summary"]["chunk_count"] = len(chunks)
        res["summary"]["chunk_size"] = size
        res["summary"]["chunk_overlap"] = overlap
        print(f" Done | Recall@{top_k}: {res['summary']['recall_at_k']*100:.1f}% | Precision@{top_k}: {res['summary']['precision_at_k']*100:.1f}%")
        chunking_results[name] = res

        # Cleanup temp db
        try:
            if temp_db.exists():
                shutil.rmtree(temp_db)
        except PermissionError:
            pass

    return chunking_results


def print_ablation_markdown_reports(
    pipeline_res: Dict[str, Any],
    chunking_res: Dict[str, Any],
    top_k: int = 4,
):
    """Formats and prints executive summary Markdown tables for both experiments."""
    print("\n" + "=" * 85)
    print("                  [DOCUMIND AI - PHASE 2 ABLATION REPORT]")
    print("=" * 85)

    # 1. Pipeline Ablation Table
    print("\n### 1. Retrieval Pipeline Ablation Study (K = 4)\n")
    pipe_rows = []
    for name, data in pipeline_res.items():
        s = data["summary"]
        pipe_rows.append({
            "Retrieval Configuration": name,
            f"Hit Rate@{top_k}": f"{s['hit_rate']*100:.1f}%",
            f"Recall@{top_k}": f"{s['recall_at_k']*100:.1f}%",
            f"Precision@{top_k}": f"{s['precision_at_k']*100:.1f}%",
            "MRR": f"{s['mrr']:.4f}",
            "Context Precision": f"{s['context_precision']:.4f}",
            "Avg Latency": f"{s['avg_latency_ms']:.1f} ms",
        })
    df_pipe = pd.DataFrame(pipe_rows)
    try:
        print(df_pipe.to_markdown(index=False))
    except Exception:
        print(df_pipe.to_string(index=False))

    # 2. Chunking Strategy Table
    print("\n### 2. Document Chunking Strategy Ablation (Full Pipeline, K = 4)\n")
    chunk_rows = []
    for name, data in chunking_res.items():
        s = data["summary"]
        chunk_rows.append({
            "Chunking Configuration": name,
            "Total Chunks": s["chunk_count"],
            f"Recall@{top_k}": f"{s['recall_at_k']*100:.1f}%",
            f"Precision@{top_k}": f"{s['precision_at_k']*100:.1f}%",
            "MRR": f"{s['mrr']:.4f}",
            "Context Precision": f"{s['context_precision']:.4f}",
            "Avg Latency": f"{s['avg_latency_ms']:.1f} ms",
        })
    df_chunk = pd.DataFrame(chunk_rows)
    try:
        print(df_chunk.to_markdown(index=False))
    except Exception:
        print(df_chunk.to_string(index=False))

    print("\n" + "=" * 85 + "\n")


def main():
    print("=== Step 8: Starting DocuMindAI Phase 2 Ablation Benchmark Suite ===")
    
    # 1. Locate sample documents
    sample_dir = current_dir / "sample_data"
    sample_files = sorted(list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.pdf")))
    if not sample_files:
        print(f"[ERROR] No sample documents found in {sample_dir.resolve()}. Run generate_sample_data.py first.")
        sys.exit(1)

    # 2. Load dataset
    eval_items = load_benchmark_dataset()
    print(f"[INIT] Loaded benchmark dataset: {len(eval_items)} evaluation questions.")

    top_k = 4

    # 3. Run Experiments
    pipeline_res = run_pipeline_ablation(sample_files, eval_items, top_k=top_k)
    chunking_res = run_chunking_ablation(sample_files, eval_items, top_k=top_k)

    # 4. Print Executive Markdown Summary
    print_ablation_markdown_reports(pipeline_res, chunking_res, top_k=top_k)

    # 5. Save Complete JSON Results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = current_dir / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ablation_run_{timestamp}.json"

    export_payload = {
        "timestamp": timestamp,
        "evaluation_dataset_size": len(eval_items),
        "top_k": top_k,
        "pipeline_ablation": {name: d["summary"] for name, d in pipeline_res.items()},
        "chunking_ablation": {name: d["summary"] for name, d in chunking_res.items()},
        "detailed_pipeline_results": {name: d["detailed_records"] for name, d in pipeline_res.items()},
        "detailed_chunking_results": {name: d["detailed_records"] for name, d in chunking_res.items()},
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_payload, f, indent=2)

    print(f"[SUCCESS] Ablation experiment results saved to: {out_file.resolve()}")


if __name__ == "__main__":
    main()
