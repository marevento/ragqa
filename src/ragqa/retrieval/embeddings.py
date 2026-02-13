"""Embedding generation using Ollama."""

from functools import lru_cache

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


@lru_cache(maxsize=1000)
def get_embedding_cached(text: str) -> tuple[float, ...]:
    """Cached embedding - returns tuple for hashability.

    Use this for repeated queries on the same text, such as titles
    that are frequently compared. The cache stores up to 1000 embeddings.

    Args:
        text: The text to embed.

    Returns:
        Tuple of embedding values (hashable for caching).
    """
    return tuple(get_embedding(text))


def clear_embedding_cache() -> None:
    """Clear the embedding cache.

    Call this when the embedding model changes or to free memory.
    """
    get_embedding_cached.cache_clear()


async def get_embedding_async(text: str) -> list[float]:
    """Generate embedding for a single text using Ollama (async).

    Args:
        text: The text to embed.

    Returns:
        List of embedding values.

    Raises:
        LLMError: If connection fails or API returns an error.
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/embed"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
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


async def get_embeddings_batch_async(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for multiple texts (async).

    Args:
        texts: List of texts to embed.

    Returns:
        List of embedding vectors.

    Raises:
        LLMError: If connection fails or API returns an error.
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/embed"

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
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
