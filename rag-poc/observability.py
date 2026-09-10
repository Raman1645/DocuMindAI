"""
observability.py - Production Latency Tracing & Pipeline Telemetry.

Measures millisecond-accurate timing across all RAG pipeline stages:
- Query Rewriting
- Hybrid Retrieval (Dense + BM25 RRF)
- Cross-Encoder Reranking
- Confidence Gate
- LLM Generation
- Evidence Verification
"""

import time
from typing import Dict, Any, Optional
from contextlib import contextmanager


class PipelineTracer:
    """Tracks latency, stage timestamps, and telemetry metadata for a RAG execution."""

    def __init__(self, query: str):
        self.query = query
        self.start_time = time.perf_counter()
        self.stages: Dict[str, float] = {}
        self.metadata: Dict[str, Any] = {}

    @contextmanager
    def trace_stage(self, stage_name: str):
        """Context manager to measure execution time of a specific pipeline stage."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
            self.stages[stage_name] = elapsed_ms

    def record_stage(self, stage_name: str, duration_ms: float) -> None:
        """Directly records a stage latency in milliseconds."""
        self.stages[stage_name] = round(duration_ms, 2)

    def set_metadata(self, key: str, value: Any) -> None:
        """Attaches diagnostic metadata (e.g. chunk count, model name, provider)."""
        self.metadata[key] = value

    def get_summary(self) -> Dict[str, Any]:
        """Returns a structured summary of the execution trace."""
        total_latency_ms = round((time.perf_counter() - self.start_time) * 1000, 2)
        return {
            "query": self.query,
            "total_latency_ms": total_latency_ms,
            "stages_ms": self.stages,
            "metadata": self.metadata,
        }

    def format_trace_box(self) -> str:
        """Formats a clean ASCII telemetry box for terminal or UI display."""
        summary = self.get_summary()
        lines = [
            "+------------------------------------------------------------+",
            f"| RAG Execution Trace: {summary['total_latency_ms']:.1f} ms total",
            "+------------------------------------------------------------+",
        ]
        for stage, ms in summary["stages_ms"].items():
            stage_display = stage.replace("_", " ").title()
            lines.append(f"|  - {stage_display:<26}: {ms:>8.1f} ms")
        
        for k, v in summary["metadata"].items():
            lines.append(f"|  * {k:<26}: {str(v):>8}")
            
        lines.append("+------------------------------------------------------------+")
        return "\n".join(lines)
