"""Embedding generation using Ollama."""

import httpx

from ragqa.config import get_settings
from ragqa.exceptions import LLMError


def get_embedding(text: str) -> list[float]:
    """Generate embedding for a single text using Ollama."""
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/embed"

    try:
        response = httpx.post(
            url,
            json={"model": settings.ollama_embed_model, "input": text},
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        # Handle both single embedding and batch response
        embeddings = data.get("embeddings", [])
        if embeddings:
            return list(embeddings[0])
        # Fallback for older API format
        return list(data.get("embedding", []))
    except httpx.ConnectError as e:
        raise LLMError(
            message="Cannot connect to Ollama. Is it running?",
            details=f"Connection refused: {url}. Run 'ollama serve' to start.",
        ) from e
    except httpx.HTTPStatusError as e:
        raise LLMError(
            message=f"Ollama embedding error: {e.response.status_code}",
            details=str(e),
        ) from e
    except Exception as e:
        raise LLMError(
            message="Failed to generate embedding",
            details=str(e),
        ) from e


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts."""
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/embed"

    try:
        response = httpx.post(
            url,
            json={"model": settings.ollama_embed_model, "input": texts},
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()
        embeddings = data.get("embeddings", [])
        return [list(emb) for emb in embeddings]
    except httpx.ConnectError as e:
        raise LLMError(
            message="Cannot connect to Ollama. Is it running?",
            details=f"Connection refused: {url}. Run 'ollama serve' to start.",
        ) from e
    except httpx.HTTPStatusError as e:
        raise LLMError(
            message=f"Ollama embedding error: {e.response.status_code}",
            details=str(e),
        ) from e
    except Exception as e:
        raise LLMError(
            message="Failed to generate embeddings",
            details=str(e),
        ) from e


def check_ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def check_model_available(model: str) -> bool:
    """Check if a specific model is available in Ollama."""
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        if response.status_code != 200:
            return False
        data = response.json()
        models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
        return model.split(":")[0] in models
    except Exception:
        return False
