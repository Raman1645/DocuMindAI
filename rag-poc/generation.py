"""
generation.py - LangChain LCEL Generation Chain & Citation Prompting Module.

Constructs an LCEL chain enforcing strict grounded answer generation
with inline source citations ([Source 1], [Source 2]) and an explicit fallback statement.
"""

import os
import re
from typing import List, Tuple, Dict, Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from config import DEFAULT_LLM_MODEL, DEFAULT_LLM_TEMPERATURE


def get_llm(model_name: str = DEFAULT_LLM_MODEL, temperature: float = DEFAULT_LLM_TEMPERATURE):
    """
    Initializes the LLM instance.
    Supports Groq (ChatGroq) via GROQ_API_KEY and OpenAI (ChatOpenAI) via OPENAI_API_KEY.
    """
    # Load .env variables if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
        
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    target_model = os.getenv("LLM_MODEL_NAME", model_name)
    
    if groq_api_key:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=target_model,
                temperature=temperature,
                groq_api_key=groq_api_key,
            )
        except ImportError:
            print("[WARNING] 'langchain-groq' package not found. Run 'pip install langchain-groq' to use Groq API.")
            
    if openai_api_key:
        return ChatOpenAI(
            model=target_model if "gpt" in target_model else "gpt-3.5-turbo",
            temperature=temperature,
            api_key=openai_api_key,
        )
        
    # Fallback Runnable Mock LLM for offline testing if no API key or package is present
    from langchain_core.runnables import RunnableLambda
    
    def mock_predict(prompt_input):
        return (
            "Based on the provided policy document, all full-time employees are eligible for a "
            "one-time home office setup stipend of $500 to purchase ergonomic equipment [Source 1]. "
            "Additionally, remote workers must use the corporate VPN when accessing customer data [Source 2]."
        )
        
    return RunnableLambda(mock_predict)


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
        "You are a factual, precise Document Q&A assistant.\n"
        "Answer the user's question STRICTLY using only the provided context chunks below.\n\n"
        "Rules:\n"
        "1. Base your answer ONLY on facts directly stated in the context. Do NOT use outside knowledge.\n"
        "2. If the context does not contain sufficient information to answer the question, state clearly:\n"
        '   "I don\'t know based on the provided documents."\n'
        "3. For every claim or factual statement in your answer, cite the supporting source chunk(s)\n"
        "   using inline brackets, e.g., [Source 1] or [Source 2].\n"
        "4. Consider recent conversation history for context when resolving pronouns or follow-up questions.\n"
        "5. Keep the answer concise, professional, and directly focused on the user's query.\n\n"
        "Context Chunks:\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    return prompt


def generate_answer(
    query: str,
    retrieved_chunks: List[Tuple[Document, float]],
    chat_history: List[Dict[str, Any]] = None,
    model_name: str = DEFAULT_LLM_MODEL,
) -> Dict[str, Any]:
    """
    Executes the generation chain using retrieved chunks, conversation history, and prompt template.
    
    Args:
        query: User question string.
        retrieved_chunks: Top-N reranked (Document, score) pairs from retrieval module.
        chat_history: List of previous conversation message dicts.
        model_name: Name of the target LLM.
        
    Returns:
        Dict[str, Any]: Dictionary containing 'answer' text and 'sources' metadata list.
    """
    formatted_context = format_context_chunks(retrieved_chunks)
    formatted_history = convert_chat_history(chat_history)
    prompt_template = get_rag_prompt_template()
    llm = get_llm(model_name)
    
    # Check if we are using LCEL or Mock fallback
    if hasattr(llm, "invoke"):
        chain = prompt_template | llm | StrOutputParser()
        answer_text = chain.invoke({
            "context": formatted_context,
            "chat_history": formatted_history,
            "question": query,
        })
    else:
        answer_text = llm.invoke(formatted_context)
        
    # Clean up internal reasoning traces (e.g. <think>...</think>) from reasoning models
    answer_text = re.sub(r"<think>.*?</think>", "", str(answer_text), flags=re.DOTALL).strip()
        
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
        
    return {
        "answer": answer_text,
        "sources": sources,
        "context_str": formatted_context,
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
