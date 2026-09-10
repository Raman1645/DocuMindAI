"""
generation.py - LangChain LCEL Generation Chain & Citation Prompting Module.

Constructs an LCEL chain enforcing strict grounded answer generation
with inline source citations ([Source 1], [Source 2]) and an explicit fallback statement.
"""

import os
import re
from typing import List, Tuple, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    CONFIDENCE_THRESHOLD,
    ENABLE_ANSWER_VERIFIER,
)
from llm_factory import create_llm
from confidence import check_retrieval_confidence
from verifier import verify_grounded_answer


def get_llm(
    model_name: str = DEFAULT_LLM_MODEL,
    provider: Optional[str] = None,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = 350,
):
    """
    Initializes the LLM instance using the pluggable llm_factory.
    Supports Groq, Ollama (local), OpenAI, and deterministic Mock fallback.
    """
    return create_llm(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def format_context_chunks(retrieved_chunks: List[Tuple[Document, float]]) -> str:
    """
    Formats retrieved candidate chunks into a structured context string for the prompt.
    """
    if not retrieved_chunks:
        return "No relevant context chunks found."
        
    formatted_blocks = []
    for idx, (doc, _score) in enumerate(retrieved_chunks, start=1):
        source_name = doc.metadata.get("source", "unknown")
        page_num = doc.metadata.get("page", 1)
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{idx}")
        
        block = (
            f"[Source {idx}]\n"
            f"File: {source_name} | Page: {page_num} | Chunk ID: {chunk_id}\n"
            f"Content:\n{doc.page_content.strip()}\n"
        )
        formatted_blocks.append(block)
        
    return "\n---\n".join(formatted_blocks)


def convert_chat_history(messages: List[Dict[str, Any]], max_history: int = 6) -> List[Any]:
    """
    Converts Streamlit session state message dicts to LangChain HumanMessage/AIMessage objects.
    Limits to max_history recent messages (e.g. last 3 Q&A turns) for context window efficiency.
    """
    if not messages:
        return []
        
    recent_messages = messages[-max_history:]
    formatted_messages = []
    
    for msg in recent_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            formatted_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_messages.append(AIMessage(content=content))
            
    return formatted_messages


def get_rag_prompt_template() -> ChatPromptTemplate:
    """
    Constructs the citation-enforcing RAG system prompt with conversational memory placeholder.
    """
    system_prompt = (
        "You are a precise, factual Document Q&A assistant.\n"
        "Answer the question directly and concisely using ONLY the provided context chunks below.\n\n"
        "Rules:\n"
        "1. Base your answer strictly on the facts present in the Context Chunks. Do not introduce outside information or assumptions.\n"
        "2. For every factual claim, cite the supporting context chunk using inline brackets, e.g., [Source 1] or [Source 2].\n"
        "3. When multiple context chunks provide complementary information, synthesize them into a single coherent answer.\n"
        "4. If the Context Chunks do not contain sufficient evidence to answer the question, respond with exact text:\n"
        '   "I don\'t know based on the provided documents."\n'
        "5. Do NOT include any internal thought process, prefixes, or conversational filler. Output only the final answer.\n\n"
        "Context Chunks:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    return prompt


def sanitize_model_output(raw_output: str) -> str:
    """
    Robustly cleans and sanitizes raw LLM generation outputs.
    
    Removes:
    1. Complete <think>...</think> blocks (case-insensitive, multiline).
    2. Unclosed <think>... blocks caused by token truncation.
    3. Stray closing </think> tags.
    4. Reasoning prefixes (e.g. 'Thought:', 'Thinking Process:', 'Reasoning:').
    5. Preserves valid inline citations [Source X], markdown formatting, and abstention statements.
    
    Returns:
        str: Clean, sanitized user-facing answer string.
    """
    if not raw_output:
        return "I don't know based on the provided documents."
        
    text = str(raw_output)
    
    # 1. Strip complete <think>...</think> blocks (case-insensitive)
    text = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Strip unclosed <think>... blocks (e.g. truncated generation)
    text = re.sub(r"<\s*think\s*>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # 3. Strip any stray closing tags
    text = re.sub(r"<\s*/\s*think\s*>", "", text, flags=re.IGNORECASE)
    
    # 4. Strip common reasoning prefixes at the start of output if present
    reasoning_prefix_pattern = r"^(?:thought|thinking process|internal analysis|reasoning):\s*.*?\n\n"
    text = re.sub(reasoning_prefix_pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # 5. Clean up leading/trailing whitespace
    text = text.strip()
    
    # 6. Fallback if entire output was empty or consumed by thinking
    if not text:
        text = "I don't know based on the provided documents."
        
    return text


def generate_answer(
    query: str,
    retrieved_chunks: List[Tuple[Document, float]],
    chat_history: List[Dict[str, Any]] = None,
    model_name: str = DEFAULT_LLM_MODEL,
    provider: Optional[str] = None,
    temperature: float = DEFAULT_LLM_TEMPERATURE,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """
    Executes the generation chain using retrieved chunks, conversation history, and prompt template.
    
    Args:
        query: User question string.
        retrieved_chunks: Top-N reranked (Document, score) pairs from retrieval module.
        chat_history: List of previous conversation message dicts.
        model_name: Name of the target LLM.
        provider: 'ollama', 'groq', 'openai', or 'mock'.
        temperature: Sampling temperature.
        max_tokens: Output token limit.
        
    Returns:
        Dict[str, Any]: Dictionary containing 'answer' text and 'sources' metadata list.
    """
    # Step 1: Guardrail - Confidence-Aware Abstention Check
    is_confident, max_score, abstention_msg = check_retrieval_confidence(
        retrieved_chunks, threshold=CONFIDENCE_THRESHOLD
    )
    if not is_confident:
        return {
            "answer": abstention_msg,
            "sources": [],
            "context_str": "",
            "is_confident": False,
            "max_relevance_score": max_score,
            "verification": {
                "is_verified": True,
                "status": "abstained_low_confidence",
                "total_citations": 0,
                "valid_citations": 0,
            },
        }

    # Step 2: Generation via LCEL
    formatted_context = format_context_chunks(retrieved_chunks)
    formatted_history = convert_chat_history(chat_history)
    prompt_template = get_rag_prompt_template()
    llm = get_llm(model_name=model_name, provider=provider, temperature=temperature, max_tokens=max_tokens)
    
    try:
        if hasattr(llm, "invoke"):
            chain = prompt_template | llm | StrOutputParser()
            raw_output = chain.invoke({
                "context": formatted_context,
                "chat_history": formatted_history,
                "question": query,
            })
        else:
            raw_output = llm.invoke(formatted_context)
            
        answer_text = sanitize_model_output(raw_output)
    except Exception as e:
        err_msg = str(e)
        active_prov = str(provider or "default")
        active_mod = str(model_name or "default")
        print(f"[GENERATION ERROR] Provider: {active_prov}, Model: {active_mod}, Details: {err_msg}")
        
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg or "rate limit" in err_msg.lower():
            answer_text = (
                f"⚠️ **Rate limit encountered on {active_prov.upper()} ({active_mod}).**\n\n"
                f"Please switch the model in the sidebar to **Groq (`llama-3.3-70b-versatile`)** or **Mock** for instant generation."
            )
        else:
            answer_text = (
                f"⚠️ **Provider Error ({active_prov.upper()} / {active_mod}):** {err_msg[:120]}"
            )
        
        return {
            "answer": answer_text,
            "sources": [],
            "context_str": formatted_context,
            "is_confident": False,
            "max_relevance_score": max_score,
            "verification": {"is_verified": False, "status": "provider_error"},
        }
        
    # Build structured sources metadata list for UI presentation
    sources = []
    for idx, (doc, score) in enumerate(retrieved_chunks, start=1):
        sources.append({
            "source_tag": f"[Source {idx}]",
            "file_name": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 1),
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "score": score,
            "content": doc.page_content,
        })
        
    # Step 3: Guardrail - Post-Generation Verification
    verification_report = {}
    if ENABLE_ANSWER_VERIFIER:
        verification_report = verify_grounded_answer(answer_text, retrieved_chunks)
        
    return {
        "answer": answer_text,
        "sources": sources,
        "context_str": formatted_context,
        "is_confident": True,
        "max_relevance_score": max_score,
        "verification": verification_report,
    }


if __name__ == "__main__":
    import shutil
    from pathlib import Path
    from ingestion import ingest_file
    from embeddings import add_documents_to_vector_store
    from retrieval import build_bm25_index, retrieve_and_rerank
    
    print("=== Step 5: LangChain Generation Chain & Citation Test ===")
    
    sample_dir = Path(__file__).parent / "sample_data"
    txt_file = sample_dir / "company_policy.txt"
    pdf_file = sample_dir / "employee_handbook.pdf"
    
    test_db_dir = Path("./chroma_gen_test_db")
    try:
        if test_db_dir.exists():
            shutil.rmtree(test_db_dir)
    except PermissionError:
        pass
        
    all_chunks = []
    if txt_file.exists():
        all_chunks.extend(ingest_file(txt_file))
    if pdf_file.exists():
        all_chunks.extend(ingest_file(pdf_file))
        
    add_documents_to_vector_store(
        all_chunks, persist_directory=str(test_db_dir), collection_name="gen_test"
    )
    bm25_idx, _ = build_bm25_index(all_chunks)
    
    test_query = "What is the equipment stipend policy?"
    print(f"\nQuery: '{test_query}'")
    
    # 1. Retrieve top reranked chunks
    retrieved_chunks = retrieve_and_rerank(
        test_query, all_chunks, bm25_idx, top_k_dense=4, top_k_bm25=4, top_n_rerank=2,
        persist_directory=str(test_db_dir), collection_name="gen_test"
    )
    
    # 2. Generate grounded answer
    response = generate_answer(test_query, retrieved_chunks)
    
    print("\n--- Generated Answer with Citations ---")
    print(response["answer"])
    
    print("\n--- Source References ---")
    for src in response["sources"]:
        print(f"{src['source_tag']} {src['file_name']} (Page {src['page']}) - Chunk ID: {src['chunk_id']}")
        
    try:
        if test_db_dir.exists():
            shutil.rmtree(test_db_dir)
    except PermissionError:
        pass
    print("\nGeneration Chain Test Completed Successfully.")
