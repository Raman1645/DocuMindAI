"""
ingestion.py - Document Loading and Chunking Module.

Handles loading PDF and TXT files using LangChain loaders and splitting them
into semantically manageable chunks via RecursiveCharacterTextSplitter.
Enriches chunk metadata with source filenames, page numbers, and unique chunk IDs.
"""

import os
from pathlib import Path
from typing import List, Union, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_document(file_path: Union[str, Path]) -> List[Document]:
    """
    Loads a single document file (PDF or TXT) and returns raw LangChain Document objects.
    
    Args:
        file_path: Absolute or relative path to the PDF or TXT file.
        
    Returns:
        List[Document]: List of parsed Document objects containing raw text and metadata.
        
    Raises:
        ValueError: If file format is not supported.
        FileNotFoundError: If the specified file does not exist.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = file_path.suffix.lower()
    
    if ext == ".pdf":
        # PyPDFLoader parses page by page, adding 'source' and 'page' (0-indexed) to metadata
        loader = PyPDFLoader(str(file_path))
        documents = loader.load()
    elif ext == ".txt":
        # TextLoader parses full file into a single document object
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()
    else:
        raise ValueError(f"Unsupported file extension '{ext}'. Only .pdf and .txt files are supported.")
        
    return documents


import time
from deduplication import compute_file_hash, DocumentRegistry


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    doc_hash: Optional[str] = None,
) -> List[Document]:
    """
    Splits loaded documents into smaller overlapping chunks using RecursiveCharacterTextSplitter.
    Attaches standardized metadata ('source', 'page', 'chunk_index', 'chunk_id', 'doc_hash', 'char_count') to every chunk.
    
    Args:
        documents: List of raw loaded Document objects.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        doc_hash: Optional SHA-256 hash of the parent file.
        
    Returns:
        List[Document]: List of chunked Document objects ready for vector embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True
    )
    
    raw_chunks = splitter.split_documents(documents)
    processed_chunks: List[Document] = []
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    for idx, chunk in enumerate(raw_chunks):
        source_name = Path(chunk.metadata.get("source", "unknown")).name
        raw_page = chunk.metadata.get("page", 0)
        page_num = raw_page + 1 if isinstance(raw_page, int) else 1
        
        # Construct deterministic unique identifier for every chunk incorporating hash if present
        if doc_hash:
            chunk_id = f"{source_name}_{doc_hash[:8]}_p{page_num}_c{idx}"
        else:
            chunk_id = f"{source_name}_p{page_num}_c{idx}"
        
        updated_metadata = {
            **chunk.metadata,
            "source": source_name,
            "file_name": source_name,
            "page": page_num,
            "chunk_index": idx,
            "chunk_id": chunk_id,
            "doc_hash": doc_hash or "unhashed",
            "char_count": len(chunk.page_content),
            "ingested_at": timestamp,
        }
        
        processed_chunks.append(
            Document(page_content=chunk.page_content, metadata=updated_metadata)
        )
        
    return processed_chunks


def ingest_file(
    file_path: Union[str, Path],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Loads document, computes SHA-256 hash, and splits into metadata-annotated chunks.
    
    Args:
        file_path: Path to the target PDF/TXT document.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        List[Document]: List of processed chunks.
    """
    file_path = Path(file_path).resolve()
    doc_hash = compute_file_hash(file_path) if file_path.exists() else None
    raw_docs = load_document(file_path)
    chunks = chunk_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap, doc_hash=doc_hash)
    return chunks


def ingest_file_idempotent(
    file_path: Union[str, Path],
    registry: Optional[DocumentRegistry] = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Tuple[List[Document], bool]:
    """
    Idempotently ingests a file.
    If the file hash has already been registered, returns ([], False) to prevent duplicate indexing.
    Otherwise parses the document, registers the hash, and returns (chunks, True).
    
    Returns:
        Tuple[List[Document], bool]: (chunks_list, was_newly_indexed).
    """
    file_path = Path(file_path).resolve()
    if registry is None:
        registry = DocumentRegistry()

    doc_hash = compute_file_hash(file_path)
    if registry.is_indexed(doc_hash):
        return [], False

    chunks = ingest_file(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    registry.register(
        doc_hash=doc_hash,
        file_name=file_path.name,
        chunk_count=len(chunks),
    )
    return chunks, True


if __name__ == "__main__":
    import sys
    
    print("=== Ingestion & Chunking Module Test ===")
    
    # Check for sample_data folder documents
    sample_dir = Path(__file__).parent / "sample_data"
    txt_sample = sample_dir / "company_policy.txt"
    pdf_sample = sample_dir / "employee_handbook.pdf"
    
    # If custom argument provided, test single file; otherwise test both sample TXT and PDF
    test_files = [Path(sys.argv[1])] if len(sys.argv) > 1 else [txt_sample, pdf_sample]
    
    for target_file in test_files:
        if not target_file.exists():
            print(f"\n[SKIP] Sample file not found: {target_file}")
            continue
            
        print(f"\n==========================================")
        print(f"Testing ingestion on: {target_file.name}")
        print(f"==========================================")
        
        chunks = ingest_file(target_file)
        print(f"Successfully generated {len(chunks)} chunks (using CHUNK_SIZE={CHUNK_SIZE}, CHUNK_OVERLAP={CHUNK_OVERLAP}).")
        
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---")
            print(f"ID:           {chunk.metadata['chunk_id']}")
            print(f"Source:       {chunk.metadata['source']}")
            print(f"Page Number:  {chunk.metadata['page']}")
            print(f"Start Index:  {chunk.metadata.get('start_index', 'N/A')}")
            print(f"Char Count:   {len(chunk.page_content)}")
            print(f"Snippet:      {repr(chunk.page_content[:120])}...")

