"""
test_ragas.py - Unit and Integration Tests for Ragas Evaluation Suite.

Validates that:
1. Benchmark dataset loading and stratified subset selection work accurately.
2. Ragas metric computation (Faithfulness, Relevancy, Precision, Recall) returns valid bounded scores.
3. Unanswerable question abstention metrics are correctly computed.
"""

import pytest
from ragas_evaluate import (
    load_benchmark_dataset,
    select_representative_subset,
    compute_ragas_metrics,
    GROQ_TARGET_MODELS,
)


def test_groq_target_models_configured():
    """Validates that exactly the 3 requested Groq models are targeted."""
    model_names = [m["model_name"] for m in GROQ_TARGET_MODELS]
    assert len(model_names) == 3
    assert "openai/gpt-oss-120b" in model_names
    assert "openai/gpt-oss-20b" in model_names
    assert "qwen/qwen3.6-27b" in model_names


def test_select_representative_subset():
    """Tests stratified subset sampling."""
    dataset = load_benchmark_dataset()
    assert len(dataset) >= 50

    subset_50 = select_representative_subset(dataset, subset_size=50)
    assert len(subset_50) == 50

    # Verify categories are represented
    cats = {it.get("category") for it in subset_50}
    assert len(cats) >= 5


def test_compute_ragas_metrics_faithful():
    """Tests Ragas metrics for a grounded, faithful answer."""
    question = "What is the core working hours policy?"
    answer = "Core hours are 10:00 AM to 4:00 PM Monday through Friday [Source 1]."
    contexts = [
        "All full-time employees must be available during core working hours from 10:00 AM to 4:00 PM Monday through Friday.",
        "Employees may take flexible hours outside core hours."
    ]
    ground_truth = "Core working hours are 10:00 AM to 4:00 PM."

    metrics = compute_ragas_metrics(question, answer, contexts, ground_truth, is_unanswerable=False)
    assert 0.0 <= metrics["faithfulness"] <= 1.0
    assert 0.0 <= metrics["answer_relevancy"] <= 1.0
    assert 0.0 <= metrics["context_precision"] <= 1.0
    assert 0.0 <= metrics["context_recall"] <= 1.0
    assert metrics["faithfulness"] > 0.6
    assert metrics["context_recall"] > 0.6


def test_compute_ragas_metrics_unanswerable():
    """Tests Ragas metrics for an unanswerable query with proper abstention."""
    question = "What is the dog walking subsidy in Seattle?"
    answer = "I don't know based on the provided documents."
    contexts = ["Company offers home office setup allowance of $500."]
    ground_truth = "I don't know based on the provided documents."

    metrics = compute_ragas_metrics(question, answer, contexts, ground_truth, is_unanswerable=True)
    assert metrics["faithfulness"] == 1.0
    assert metrics["context_recall"] == 1.0
