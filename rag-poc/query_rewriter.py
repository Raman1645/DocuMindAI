"""
query_rewriter.py - Conversational Query Rewriting & Disambiguation Module.

Translates pronoun-heavy, ambiguous conversational follow-up questions
(e.g., "What about the second one?") into fully self-contained search queries
based on prior conversation history before passing to the hybrid retrieval pipeline.
"""

import re
from typing import List, Dict, Any, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from llm_factory import create_llm


# Single-word pronoun & reference triggers
SINGLE_WORD_TRIGGERS = {
    "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "second", "third", "other", "another", "previous", "former", "latter",
}

# Multi-word conversational phrase triggers
PHRASE_TRIGGERS = [
    "what about", "how about", "and for", "why then", "what if",
    "how many more", "which one", "which of these", "tell me more",
]


def contains_conversational_reference(query: str) -> bool:
    """
    Fast heuristic check to detect if the query references prior context.
    Returns True if pronouns or follow-up phrases are detected.
    """
    q_lower = query.lower().strip()
    
    # 1. Check multi-word phrase triggers first
    if any(phrase in q_lower for phrase in PHRASE_TRIGGERS):
        return True
        
    # 2. Check single word tokens
    words = re.findall(r"\w+", q_lower)
    if any(w in SINGLE_WORD_TRIGGERS for w in words):
        return True
        
    return False


def format_history_for_rewriter(chat_history: List[Dict[str, Any]], max_turns: int = 4) -> str:
    """Formats the last N conversation turns into structured text for rewriter prompt."""
    if not chat_history:
        return ""
        
    recent = chat_history[-max_turns:]
    lines = []
    for msg in recent:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "").strip()
        lines.append(f"{role}: {content}")
        
    return "\n".join(lines)


REWRITE_PROMPT = PromptTemplate.from_template(
    "Given the conversation history and a follow-up question, rewrite the follow-up question "
    "to be a standalone search query that can be understood without the conversation history.\n\n"
    "Rules:\n"
    "1. Replace pronouns (it, they, this, that, etc.) with the actual subjects discussed in history.\n"
    "2. Do NOT answer the question. Only output the rewritten standalone question.\n"
    "3. Keep the query concise, factual, and focused on information retrieval.\n\n"
    "Conversation History:\n"
    "{chat_history}\n\n"
    "Follow-up Question: {question}\n\n"
    "Standalone Query:"
)


def rewrite_query(
    query: str,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    force_rewrite: bool = False,
) -> str:
    """
    Rewrites a conversational query into a standalone search query if needed.
    
    Args:
        query: User input search question.
        chat_history: List of prior chat message dicts.
        provider: Target LLM provider ('gemini', 'groq', 'ollama', 'mock').
        model_name: Target model identifier.
        force_rewrite: If True, bypasses heuristic and always executes rewrite chain.
        
    Returns:
        str: Standalone query string.
    """
    if not chat_history or len(chat_history) == 0:
        return query.strip()
        
    # Heuristic bypass: If self-contained and not forced, return as-is
    if not force_rewrite and not contains_conversational_reference(query):
        return query.strip()
        
    formatted_hist = format_history_for_rewriter(chat_history)
    if not formatted_hist:
        return query.strip()
        
    try:
        llm = create_llm(provider=provider, model_name=model_name, temperature=0.0, max_tokens=100)
        
        if hasattr(llm, "invoke"):
            chain = REWRITE_PROMPT | llm | StrOutputParser()
            rewritten = chain.invoke({
                "chat_history": formatted_hist,
                "question": query,
            })
            rewritten_text = re.sub(r"<think>.*?</think>", "", str(rewritten), flags=re.DOTALL).strip()
            rewritten_text = re.sub(r"^[\"']|[\"']$", "", rewritten_text).strip()
            return rewritten_text if rewritten_text else query.strip()
        else:
            return query.strip()
    except Exception as e:
        # Graceful fallback to original query if rewriter call fails
        print(f"[REWRITER WARNING] Query rewriting failed: {e}. Using original query.")
        return query.strip()
