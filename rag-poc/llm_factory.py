"""
llm_factory.py - Unified Provider Factory for LLM Generation.

Provides an extensible, model-agnostic factory interface supporting:
1. Groq Cloud (ChatGroq) via GROQ_API_KEY
2. Local Ollama (ChatOllama) via http://localhost:11434 (100% offline, zero rate limits)
3. OpenAI (ChatOpenAI) via OPENAI_API_KEY
4. Offline Mock LLM for deterministic, zero-cost unit testing

Allows seamless switching of generation backends via configuration without modifying RAG pipeline logic.
"""

import os
import json
import urllib.request
from typing import Optional, Dict, Any, List
from langchain_core.runnables import RunnableLambda

# Load .env variables if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


GROQ_SUPPORTED_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]


def normalize_groq_model_name(model_name: str) -> str:
    """Normalizes Groq model names to official endpoint identifiers."""
    m_clean = model_name.strip()
    if "120b" in m_clean.lower():
        return "openai/gpt-oss-120b"
    elif "20b" in m_clean.lower():
        return "openai/gpt-oss-20b"
    elif "qwen" in m_clean.lower() or "27b" in m_clean.lower():
        return "qwen/qwen3.6-27b"
    return m_clean


def get_default_provider_and_model() -> tuple[str, str]:
    """Resolves default provider and model from environment or defaults."""
    provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    model = os.getenv("LLM_MODEL_NAME", "").strip()

    if not provider:
        if os.getenv("GROQ_API_KEY"):
            provider = "groq"
            model = model or "openai/gpt-oss-120b"
        elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            provider = "gemini"
            model = model or "gemini-3.6-flash"
        elif os.getenv("OLLAMA_MODEL"):
            provider = "ollama"
            model = model or os.getenv("OLLAMA_MODEL", "qwen3:8b")
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
            model = model or "gpt-4o-mini"
        else:
            provider = "mock"
            model = "mock-deterministic"

    if not model:
        if provider == "groq":
            model = "openai/gpt-oss-120b"
        elif provider in ("gemini", "google"):
            model = "gemini-3.6-flash"
        elif provider == "ollama":
            model = "qwen3:8b"
        elif provider == "openai":
            model = "gpt-4o-mini"
        else:
            model = "mock-deterministic"

    return provider, model


def create_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 350,
    ollama_base_url: Optional[str] = None,
    **kwargs: Any,
):
    """
    Instantiates an LLM instance based on specified provider and model name.

    Args:
        provider: 'groq', 'ollama', 'openai', or 'mock'.
        model_name: Model identifier (e.g. 'llama-3.1-8b-instant', 'qwen3:8b', 'gpt-4o-mini').
        temperature: Sampling temperature (0.0 for deterministic factual Q&A).
        max_tokens: Maximum response tokens.
        ollama_base_url: Custom Ollama URL (defaults to http://localhost:11434).

    Returns:
        LangChain compatible ChatModel or Runnable instance.
    """
    default_provider, default_model = get_default_provider_and_model()
    target_provider = (provider or default_provider).lower().strip()
    target_model = model_name or default_model

    # 1. Local Ollama Provider (100% local, no rate limits)
    if target_provider == "ollama":
        base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        num_gpu = int(os.getenv("OLLAMA_NUM_GPU", "0"))
        try:
            from langchain_ollama import ChatOllama
            return ChatOllama(
                model=target_model,
                base_url=base_url,
                temperature=temperature,
                num_predict=max_tokens,
                num_gpu=num_gpu,
                **kwargs,
            )
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama
                return ChatOllama(
                    model=target_model,
                    base_url=base_url,
                    temperature=temperature,
                    num_predict=max_tokens,
                    num_gpu=num_gpu,
                    **kwargs,
                )
            except ImportError:
                raise ImportError(
                    "Ollama package not found. Run: pip install langchain-ollama"
                )

    # 2. Groq Cloud Provider
    elif target_provider == "groq":
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required for Groq provider.")
        try:
            from langchain_groq import ChatGroq
            normalized_groq_model = normalize_groq_model_name(target_model)
            groq_tokens = min(max_tokens, 512) if "qwen" in normalized_groq_model.lower() else max_tokens
            return ChatGroq(
                model_name=normalized_groq_model,
                temperature=temperature,
                max_tokens=groq_tokens,
                max_retries=3,
                groq_api_key=groq_api_key,
                **kwargs,
            )
        except ImportError:
            raise ImportError("Groq package not found. Run: pip install langchain-groq")

    # 3. Google Gemini Provider
    elif target_provider in ("gemini", "google"):
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required for Gemini.")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            # Normalize model name to active Google endpoint
            gemini_model = target_model if "gemini" in target_model else "gemini-3.6-flash"
            if gemini_model in ("gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"):
                gemini_model = "gemini-3.6-flash"
            return ChatGoogleGenerativeAI(
                model=gemini_model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=gemini_api_key,
                **kwargs,
            )
        except ImportError:
            raise ImportError("Google GenAI package not found. Run: pip install langchain-google-genai")

    # 4. OpenAI Cloud Provider
    elif target_provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI provider.")
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=target_model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=openai_api_key,
                **kwargs,
            )
        except ImportError:
            raise ImportError("OpenAI package not found. Run: pip install langchain-openai")

    # 5. Mock Provider (Offline testing fallback)
    elif target_provider == "mock":
        def mock_predict(prompt_input):
            # Extract question if dict input
            q = prompt_input.get("question", "") if isinstance(prompt_input, dict) else str(prompt_input)
            return (
                f"Based on the provided documents, all full-time employees are eligible for home office benefits [Source 1]. "
                f"For additional details, please consult company policies [Source 2]."
            )
        return RunnableLambda(mock_predict)

    else:
        raise ValueError(
            f"Unsupported LLM provider '{target_provider}'. Supported providers: 'gemini', 'groq', 'ollama', 'openai', 'mock'."
        )


def list_supported_providers() -> Dict[str, Dict[str, Any]]:
    """Returns catalog of supported providers and typical model options."""
    return {
        "gemini": {
            "name": "Google Gemini",
            "models": ["gemini-1.5-flash", "gemini-2.5-flash", "gemini-1.5-pro"],
            "requires_api_key": True,
            "rate_limited": True,
        },
        "groq": {
            "name": "Groq Cloud API",
            "models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "qwen/qwen3.6-27b"],
            "requires_api_key": True,
            "rate_limited": True,
        },
        "ollama": {
            "name": "Ollama (Local / Offline)",
            "models": ["qwen3:8b", "qwen2.5:8b", "qwen2.5:3b", "llama3.2:3b", "llama3.2:1b", "mistral:7b"],
            "requires_api_key": False,
            "rate_limited": False,
        },
        "openai": {
            "name": "OpenAI API",
            "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            "requires_api_key": True,
            "rate_limited": True,
        },
        "mock": {
            "name": "Deterministic Mock (Testing)",
            "models": ["mock-deterministic"],
            "requires_api_key": False,
            "rate_limited": False,
        },
    }
