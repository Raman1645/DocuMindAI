"""
model_benchmark.py - Multi-Model Comparative Benchmark Suite for Fixed RAG Retrieval.

Benchmarks multiple LLMs under an identical, fixed hybrid retrieval pipeline:
1. Ingests sample documents & retrieves top-4 candidate chunks via Hybrid + Cross-Encoder.
2. Evaluates multiple target models:
   - Ollama Local Models (e.g. qwen3:8b, qwen2.5:8b, llama3.2:3b)
   - Groq Cloud Models (e.g. llama-3.1-8b-instant, llama-3.3-70b-versatile, qwen/qwen3.6-27b)
   - OpenAI Models (e.g. gpt-4o-mini)
3. Computes comparative generation metrics:
   - Groundedness / Factual Recall (%)
   - Citation Accuracy (%)
   - Correct Abstention Rate on Unanswerables (%)
   - Generation Latency (ms) and Total Pipeline Latency (ms)
4. Outputs formatted Markdown comparison table and saves JSON results to evaluation/results/.
"""

import json
import os
import sys
import shutil
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd

# Add rag-poc to python path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from ingestion import ingest_file
from embeddings import add_documents_to_vector_store
from retrieval import build_bm25_index, retrieve_by_mode
from generation import generate_answer
from llm_factory import create_llm, list_supported_providers
from evaluation.generation_metrics import (
    compute_groundedness_score,
    compute_citation_accuracy,
    compute_abstention_accuracy,
)


def load_eval_dataset(dataset_path: str = None) -> List[Dict[str, Any]]:
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


def prepare_benchmark_retrieval(
    eval_items: List[Dict[str, Any]],
    top_k: int = 4,
) -> Tuple[List[Dict[str, Any]], Path]:
    """
    Ingests sample documents once and pre-retrieves candidate chunks for every question.
    Ensures 100% identical retrieval context for all compared models.
    """
    sample_dir = current_dir / "sample_data"
    sample_files = sorted(list(sample_dir.glob("*.txt")) + list(sample_dir.glob("*.pdf")))
    if not sample_files:
        raise FileNotFoundError(f"No sample documents found in {sample_dir.resolve()}.")

    print(f"\n[INIT] Indexing {len(sample_files)} sample documents for multi-model benchmark...")
    temp_db_dir = current_dir / "chroma_model_benchmark_db"
    try:
        if temp_db_dir.exists():
            shutil.rmtree(temp_db_dir)
    except PermissionError:
        pass

    all_chunks = []
    for file_path in sample_files:
        chunks = ingest_file(file_path)
        all_chunks.extend(chunks)

    add_documents_to_vector_store(
        all_chunks,
        persist_directory=str(temp_db_dir),
        collection_name="model_benchmark",
    )
    bm25_idx, _ = build_bm25_index(all_chunks)
    print(f"[INIT] Corpus indexed: {len(all_chunks)} chunks ready.")

    print(f"[INIT] Pre-retrieving top-{top_k} hybrid chunks for {len(eval_items)} questions...")
    retrieval_records = []
    for item in eval_items:
        t0 = time.perf_counter()
        retrieved_chunks = retrieve_by_mode(
            mode="hybrid_rerank",
            query=item["question"],
            documents=all_chunks,
            bm25_index=bm25_idx,
            top_k=top_k,
            persist_directory=str(temp_db_dir),
            collection_name="model_benchmark",
        )
        ret_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        
        retrieval_records.append({
            "item": item,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_latency_ms": ret_latency_ms,
        })
        
    return retrieval_records, temp_db_dir


def evaluate_model_on_retrieved_data(
    provider: str,
    model_name: str,
    retrieval_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Executes generation for a specific model over pre-retrieved context chunks.
    """
    detailed_results = []
    gen_latencies = []
    
    # Warmup call
    try:
        _ = generate_answer(
            query="Hello",
            retrieved_chunks=retrieval_records[0]["retrieved_chunks"][:1],
            model_name=model_name,
            provider=provider,
            max_tokens=20,
        )
    except Exception as e:
        print(f"\n[WARNING] Warmup failed for {provider}/{model_name}: {e}")

    for idx, rec in enumerate(retrieval_records, start=1):
        item = rec["item"]
        retrieved_chunks = rec["retrieved_chunks"]
        ret_lat_ms = rec["retrieval_latency_ms"]
        
        qid = item.get("id", f"Q{idx:02d}")
        question = item["question"]
        expected_keywords = item.get("expected_keywords", [])
        is_answerable = item.get("is_answerable", True)

        t_gen_start = time.perf_counter()
        try:
            gen_resp = generate_answer(
                query=question,
                retrieved_chunks=retrieved_chunks,
                model_name=model_name,
                provider=provider,
                max_tokens=350,
            )
            answer_text = gen_resp.get("answer", "")
        except Exception as e:
            answer_text = f"[ERROR: Generation Failed - {str(e)}]"
        gen_latency_ms = round((time.perf_counter() - t_gen_start) * 1000, 2)
        gen_latencies.append(gen_latency_ms)

        # Calculate metrics
        groundedness = compute_groundedness_score(answer_text, retrieved_chunks, expected_keywords, is_answerable)
        citation_acc = compute_citation_accuracy(answer_text, retrieved_chunks, expected_keywords) if is_answerable else 1.0
        abstention_acc = compute_abstention_accuracy(answer_text, is_answerable)

        detailed_results.append({
            "id": qid,
            "category": item.get("category", "general"),
            "question": question,
            "is_answerable": is_answerable,
            "groundedness": groundedness,
            "citation_accuracy": citation_acc,
            "abstention_accuracy": abstention_acc,
            "retrieval_latency_ms": ret_lat_ms,
            "generation_latency_ms": gen_latency_ms,
            "total_latency_ms": round(ret_lat_ms + gen_latency_ms, 2),
            "generated_answer": answer_text,
        })
        
        if idx % 10 == 0 or idx == len(retrieval_records):
            print(f"    Progress: {idx}/{len(retrieval_records)} evaluated...", flush=True)

    answerable = [r for r in detailed_results if r["is_answerable"]]
    unanswerable = [r for r in detailed_results if not r["is_answerable"]]
    
    ans_count = len(answerable) if answerable else 1
    unans_count = len(unanswerable) if unanswerable else 1
    
    summary = {
        "provider": provider,
        "model_name": model_name,
        "total_queries": len(detailed_results),
        "mean_groundedness": round(sum(r["groundedness"] for r in answerable) / ans_count, 4),
        "mean_citation_accuracy": round(sum(r["citation_accuracy"] for r in answerable) / ans_count, 4),
        "abstention_rate": round(sum(r["abstention_accuracy"] for r in unanswerable) / unans_count, 4),
        "avg_generation_latency_ms": round(sum(gen_latencies) / len(gen_latencies), 2),
        "avg_total_latency_ms": round(sum(r["total_latency_ms"] for r in detailed_results) / len(detailed_results), 2),
    }
    
    return {
        "summary": summary,
        "detailed_results": detailed_results,
    }


def print_model_comparison_table(model_results: Dict[str, Any]):
    """Formats and prints an executive comparison Markdown table."""
    print("\n" + "=" * 85)
    print("           [DOCUMIND AI - PHASE 3 MULTI-MODEL BENCHMARK REPORT]")
    print("=" * 85)
    
    rows = []
    for label, data in model_results.items():
        s = data["summary"]
        rows.append({
            "Model Name": label,
            "Provider": s["provider"].upper(),
            "Groundedness": f"{s['mean_groundedness']*100:.1f}%",
            "Citation Acc": f"{s['mean_citation_accuracy']*100:.1f}%",
            "Abstention Acc": f"{s['abstention_rate']*100:.1f}%",
            "Gen Latency": f"{s['avg_generation_latency_ms']:.1f} ms",
            "Total Latency": f"{s['avg_total_latency_ms']:.1f} ms",
        })
        
    df = pd.DataFrame(rows)
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df.to_string(index=False))
    print("\n" + "=" * 85 + "\n")


def main():
    parser = argparse.ArgumentParser(description="DocuMindAI Multi-Model Benchmark Suite")
    parser.add_argument("--subset", type=int, default=20, help="Number of questions to benchmark (default 20 for fast evaluation, 80 for full)")
    parser.add_argument("--models", nargs="+", default=None, help="List of provider:model targets (e.g. ollama:qwen3:8b groq:llama-3.1-8b-instant)")
    args = parser.parse_args()

    print("=== Step 9: Starting DocuMindAI Phase 3 Multi-Model Benchmark Suite ===")
    
    # 1. Load dataset
    full_dataset = load_eval_dataset()
    if args.subset and args.subset < len(full_dataset):
        # Select balanced representative subset across all categories
        step = max(1, len(full_dataset) // args.subset)
        eval_items = full_dataset[::step][:args.subset]
        print(f"[DATASET] Selected balanced representative subset of {len(eval_items)} questions (from total {len(full_dataset)}).")
    else:
        eval_items = full_dataset
        print(f"[DATASET] Running full benchmark on all {len(eval_items)} questions.")

    # 2. Pre-retrieve context once
    retrieval_records, temp_db_dir = prepare_benchmark_retrieval(eval_items, top_k=4)

    # 3. Define candidate model list
    if args.models:
        candidate_models = []
        for m in args.models:
            if ":" in m:
                p, mod = m.split(":", 1)
                candidate_models.append((p.strip(), mod.strip(), f"{p.strip()}:{mod.strip()}"))
            else:
                candidate_models.append(("groq", m.strip(), f"groq:{m.strip()}"))
    else:
        # Default model roster: prioritize Gemini if GEMINI_API_KEY is present
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            candidate_models = [
                ("gemini", "gemini-3.6-flash", "Google Gemini (gemini-3.6-flash)"),
            ]
        else:
            candidate_models = [
                ("groq", "llama-3.1-8b-instant", "Groq (Llama-3.1-8B) [Cloud Fast]"),
                ("groq", "llama-3.3-70b-versatile", "Groq (Llama-3.3-70B) [Cloud High-IQ]"),
            ]

    model_benchmark_results = {}
    for provider, model_name, display_label in candidate_models:
        print(f"\n==================================================================")
        print(f"Benchmarking Model: {display_label} (Provider: {provider}, Model: {model_name})")
        print(f"==================================================================")
        try:
            res = evaluate_model_on_retrieved_data(
                provider=provider,
                model_name=model_name,
                retrieval_records=retrieval_records,
            )
            model_benchmark_results[display_label] = res
            s = res["summary"]
            print(f"  --> Completed: Groundedness={s['mean_groundedness']*100:.1f}%, Citation Acc={s['mean_citation_accuracy']*100:.1f}%, Gen Latency={s['avg_generation_latency_ms']:.1f}ms")
        except Exception as e:
            print(f"  --> [ERROR] Benchmark failed for {display_label}: {e}")

    # 4. Print comparison table
    print_model_comparison_table(model_benchmark_results)

    # 5. Save JSON report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = current_dir / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"model_benchmark_{timestamp}.json"

    export_data = {
        "timestamp": timestamp,
        "evaluation_questions_count": len(eval_items),
        "models_summary": {label: d["summary"] for label, d in model_benchmark_results.items()},
        "detailed_results": {label: d["detailed_results"] for label, d in model_benchmark_results.items()},
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)
        
    print(f"[SUCCESS] Multi-model benchmark report saved to: {out_file.resolve()}")

    # Cleanup temp db
    try:
        if temp_db_dir.exists():
            shutil.rmtree(temp_db_dir)
    except PermissionError:
        pass


if __name__ == "__main__":
    main()
