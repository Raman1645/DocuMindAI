"""
deduplication.py - Idempotent Ingestion & SHA-256 Document Hashing Registry.

Prevents duplicate embeddings and vector DB bloat by calculating SHA-256 checksums
of document content and maintaining a persistent index registry.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Union, Dict, Any, Optional, Set

from config import DATA_DIR

DEFAULT_REGISTRY_FILE = DATA_DIR / "document_registry.json"


def compute_file_hash(file_source: Union[str, Path, bytes]) -> str:
    """
    Computes a deterministic SHA-256 hexadecimal digest for a file path or raw bytes.

    Args:
        file_source: Path to file (str/Path) or raw file bytes.

    Returns:
        str: 64-character SHA-256 hexadecimal string.
    """
    sha256 = hashlib.sha256()

    if isinstance(file_source, bytes):
        sha256.update(file_source)
    else:
        path = Path(file_source)
        if not path.exists():
            raise FileNotFoundError(f"Cannot hash non-existent file: {path.resolve()}")
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)

    return sha256.hexdigest()


class DocumentRegistry:
    """Persistent JSON registry tracking indexed document hashes and metadata."""

    def __init__(self, registry_path: Optional[Union[str, Path]] = None):
        self.registry_path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_FILE
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2)
        except Exception as e:
            print(f"[REGISTRY WARNING] Failed to persist registry: {e}")

    def is_indexed(self, doc_hash: str) -> bool:
        """Returns True if the document hash is already registered."""
        return doc_hash in self.records

    def register(
        self,
        doc_hash: str,
        file_name: str,
        chunk_count: int,
        collection_name: str = "rag_documents",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Registers a newly indexed document."""
        self.records[doc_hash] = {
            "doc_hash": doc_hash,
            "file_name": file_name,
            "chunk_count": chunk_count,
            "collection_name": collection_name,
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "extra_metadata": extra_metadata or {},
        }
        self._save()

    def get_indexed_hashes(self) -> Set[str]:
        """Returns set of all registered SHA-256 hashes."""
        return set(self.records.keys())

    def clear(self) -> None:
        """Clears the registry."""
        self.records = {}
        if self.registry_path.exists():
            try:
                self.registry_path.unlink()
            except Exception:
                pass
