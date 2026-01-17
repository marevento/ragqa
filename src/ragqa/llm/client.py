"""Ollama LLM client with streaming support."""

import json
import random
from collections.abc import Generator

import httpx

from ragqa.config import get_settings
from ragqa.exceptions import LLMError


def generate(prompt: str, stream: bool = False) -> str | Generator[str, None, None]:
    """Generate text using Ollama LLM."""
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/generate"

    payload = {
        "model": settings.ollama_model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": settings.llm_temperature,
            "num_ctx": settings.llm_context_window,
            "seed": random.randint(1, 1000000),  # Prevent Ollama KV cache
        },
    }

    if stream:
        return _stream_generate(url, payload)
    else:
        return _sync_generate(url, payload)


def _sync_generate(url: str, payload: dict[str, object]) -> str:
    """Non-streaming generation."""
    try:
        response = httpx.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        return str(data.get("response", ""))
    except httpx.ConnectError as e:
        raise LLMError(
            message="Cannot connect to Ollama. Is it running?",
            details=f"Connection refused: {url}. Run 'ollama serve' to start.",
        ) from e
    except httpx.HTTPStatusError as e:
        raise LLMError(
            message=f"Ollama generation error: {e.response.status_code}",
            details=str(e),
        ) from e
    except Exception as e:
        raise LLMError(
            message="Failed to generate response",
            details=str(e),
        ) from e


def _stream_generate(
    url: str, payload: dict[str, object]
) -> Generator[str, None, None]:
    """Streaming generation."""
    try:
        with httpx.stream("POST", url, json=payload, timeout=120.0) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
    except httpx.ConnectError as e:
        raise LLMError(
            message="Cannot connect to Ollama. Is it running?",
            details=f"Connection refused: {url}. Run 'ollama serve' to start.",
        ) from e
    except httpx.HTTPStatusError as e:
        raise LLMError(
            message=f"Ollama generation error: {e.response.status_code}",
            details=str(e),
        ) from e
    except Exception as e:
        raise LLMError(
            message="Failed to generate response",
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


def get_available_models() -> list[str]:
    """Get list of available models in Ollama."""
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        if response.status_code != 200:
            return []
        data = response.json()
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []
