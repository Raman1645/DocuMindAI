"""
test_production.py - Unit and Integration Tests for Phase 5 Production Engineering.

Validates that:
1. SHA-256 document hashing is deterministic and reproducible.
2. DocumentRegistry ensures idempotent ingestion (no duplicate chunks).
3. Ingested chunk metadata contains all required production lineage fields.
4. PipelineTracer records millisecond-accurate stage latencies and formats traces.
"""

import pytest
import shutil
import time
from pathlib import Path

from deduplication import compute_file_hash, DocumentRegistry
from ingestion import ingest_file, ingest_file_idempotent
from observability import PipelineTracer


@pytest.fixture
def temp_sample_file(tmp_path):
    """Creates a temporary sample text file."""
    f = tmp_path / "sample_policy.txt"
    f.write_text("All employees must use FileVault or BitLocker disk encryption.", encoding="utf-8")
    return f


def test_compute_file_hash(temp_sample_file):
    """Tests deterministic SHA-256 hash calculation."""
    hash1 = compute_file_hash(temp_sample_file)
    hash2 = compute_file_hash(temp_sample_file)
    
    assert len(hash1) == 64
    assert hash1 == hash2

    # Verify raw byte hashing matches file hashing
    byte_hash = compute_file_hash(temp_sample_file.read_bytes())
    assert byte_hash == hash1


def test_document_registry_idempotency(tmp_path, temp_sample_file):
    """Tests that DocumentRegistry prevents duplicate file ingestion."""
    reg_file = tmp_path / "test_reg.json"
    registry = DocumentRegistry(registry_path=reg_file)
    
    # 1. First ingestion -> should be newly indexed
    chunks_1, was_new = ingest_file_idempotent(temp_sample_file, registry=registry)
    assert was_new is True
    assert len(chunks_1) > 0

    # 2. Second ingestion of identical file -> should skip
    chunks_2, was_new_second = ingest_file_idempotent(temp_sample_file, registry=registry)
    assert was_new_second is False
    assert len(chunks_2) == 0


def test_metadata_enrichment(temp_sample_file):
    """Tests that chunks contain all production metadata fields."""
    chunks = ingest_file(temp_sample_file)
    assert len(chunks) > 0
    
    chunk = chunks[0]
    meta = chunk.metadata
    
    assert "doc_hash" in meta
    assert len(meta["doc_hash"]) == 64
    assert "chunk_id" in meta
    assert "char_count" in meta
    assert meta["char_count"] == len(chunk.page_content)
    assert "ingested_at" in meta
    assert "page" in meta


def test_pipeline_tracer():
    """Tests latency tracking and trace box formatting."""
    tracer = PipelineTracer(query="What is the policy?")
    
    with tracer.trace_stage("retrieval"):
        time.sleep(0.01)  # 10ms simulated
        
    with tracer.trace_stage("generation"):
        time.sleep(0.02)  # 20ms simulated

    tracer.set_metadata("provider", "gemini")
    tracer.set_metadata("retrieved_chunks", 4)
    
    summary = tracer.get_summary()
    assert summary["total_latency_ms"] >= 30.0
    assert "retrieval" in summary["stages_ms"]
    assert "generation" in summary["stages_ms"]
    assert summary["metadata"]["provider"] == "gemini"

    trace_box = tracer.format_trace_box()
    assert "Retrieval" in trace_box
    assert "Generation" in trace_box
    assert "gemini" in trace_box
