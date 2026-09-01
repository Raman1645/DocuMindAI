"""
embeddings.py - Embedding Generation & Vector Store Management Module.

Uses HuggingFace SentenceTransformers (BAAI/bge-small-en-v1.5) to embed text chunks
and manages persistent vector storage via ChromaDB.
"""

from typing import List, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K_DENSE,
)

# Global singleton cache for embedding model to avoid reloading weights on every call
_EMBEDDING_MODEL = None


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFace BAAI/bge-small-en-v1.5 embedding function.
    Uses normalized embeddings to ensure cosine similarity scoring is exact.
    
    Returns:
        HuggingFaceEmbeddings: Configured LangChain embedding model wrapper.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}  # Crucial for BGE model cosine distance math
        
        _EMBEDDING_MODEL = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
    return _EMBEDDING_MODEL


def get_vector_store(
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """
    Initializes or opens an existing ChromaDB persistent collection.
    
    Args:
        persist_directory: Directory path where ChromaDB stores index files.
        collection_name: Name of the target Chroma collection.
        
    Returns:
        Chroma: LangChain Chroma vector store instance.
    """
    embedding_function = get_embedding_function()
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )
    return vector_store


def add_documents_to_vector_store(
    documents: List[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Chroma:
    """
    Adds chunked Document objects to ChromaDB using unique 'chunk_id' from metadata.
    Avoids duplicate insertion by checking existing document IDs.
    
    Args:
        documents: List of chunked Document objects.
        persist_directory: Storage directory path.
        collection_name: Name of Chroma collection.
        
    Returns:
        Chroma: Updated vector store instance.
    """
    vector_store = get_vector_store(persist_directory, collection_name)
    
    if not documents:
        return vector_store
        
    # Extract unique IDs generated during chunking (e.g., 'company_policy.txt_p1_c0')
    chunk_ids = [doc.metadata.get("chunk_id") for doc in documents]
    
    # Add documents to collection using chunk_ids to ensure uniqueness
    vector_store.add_documents(documents=documents, ids=chunk_ids)
    return vector_store


def dense_similarity_search(
    query: str,
    top_k: int = TOP_K_DENSE,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> List[Tuple[Document, float]]:
    """
    Executes a dense vector similarity search against the ChromaDB collection.
    
    Args:
        query: User search question string.
        top_k: Number of nearest neighbor documents to retrieve.
        persist_directory: Path to Chroma database storage.
        collection_name: Chroma collection identifier.
        
    Returns:
        List[Tuple[Document, float]]: List of (Document, similarity_score) tuples.
    """
    vector_store = get_vector_store(persist_directory, collection_name)
    
    # BGE models perform best when query string is prefixed or standard search is executed
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return results


if __name__ == "__main__":
    import shutil
    from ingestion import ingest_file
    
    print("=== Step 2: Embeddings & Chroma Vector Store Test ===")
    
    sample_dir = Path(__file__).parent / "sample_data"
    txt_file = sample_dir / "company_policy.txt"
    pdf_file = sample_dir / "employee_handbook.pdf"
    
    # 1. Clean previous ChromaDB test store if exists
    test_db_dir = Path("./chroma_test_db")
    if test_db_dir.exists():
        shutil.rmtree(test_db_dir)
        
    # 2. Load & Chunk sample files
    print("\n1. Ingesting documents...")
    all_chunks = []
    if txt_file.exists():
        all_chunks.extend(ingest_file(txt_file))
    if pdf_file.exists():
        all_chunks.extend(ingest_file(pdf_file))
        
    print(f"Total chunks created: {len(all_chunks)}")
    
    # 3. Index into ChromaDB
    print("\n2. Embedding and indexing into ChromaDB (BAAI/bge-small-en-v1.5)...")
    store = add_documents_to_vector_store(
        all_chunks, persist_directory=str(test_db_dir), collection_name="test_collection"
    )
    print("Indexing complete.")
    
    # 4. Run similarity search tests
    test_queries = [
        "What is the core working hours policy?",
        "How many days of paid leave do full-time employees get?",
        "What is the home office setup allowance?"
    ]
    
    print("\n3. Testing Dense Similarity Search:")
    for query in test_queries:
        print(f"\n==========================================")
        print(f"Query: '{query}'")
        print(f"==========================================")
        results = dense_similarity_search(
            query, top_k=2, persist_directory=str(test_db_dir), collection_name="test_collection"
        )
        
        for idx, (doc, score) in enumerate(results):
            print(f"\n--- Rank {idx+1} (Score/Distance: {score:.4f}) ---")
            print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
            print(f"Source:   {doc.metadata.get('source')} (Page {doc.metadata.get('page')})")
            print(f"Snippet:  {repr(doc.page_content[:140])}...")
            
    # Clean test directory gracefully on Windows
    try:
        if test_db_dir.exists():
            shutil.rmtree(test_db_dir)
    except PermissionError:
        pass
    print("\nTest completed cleanly.")
