"""Shared test fixtures."""

import json
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from ragqa.config import Settings, get_settings
from ragqa.core.models import Chunk, Document


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def mock_settings(temp_dir: Path) -> Generator[Settings, None, None]:
    """Provide mock settings for testing with cache cleared."""
    # Clear the settings cache before and after test
    get_settings.cache_clear()
    settings = Settings(
        ollama_base_url="http://localhost:11434",
        ollama_model="test-model",
        ollama_embed_model="test-embed",
        papers_dir=temp_dir / "papers",
        chroma_persist_dir=temp_dir / "chroma",
    )
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def mock_ollama_response() -> dict[str, object]:
    """Mock Ollama API response."""
    return {
        "response": "This is a test response about ToolMem.",
        "done": True,
    }


@pytest.fixture
def mock_embedding() -> list[float]:
    """Mock embedding vector."""
    return [0.1] * 768


@pytest.fixture
def mock_embedding_response() -> dict[str, Any]:
    """Standard embedding response from Ollama."""
    return {"embeddings": [[0.1] * 768]}


@pytest.fixture
def mock_generate_response() -> dict[str, Any]:
    """Standard LLM response from Ollama."""
    return {"response": "Test answer from LLM.", "done": True}


@pytest.fixture
def mock_httpx_client(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, Any]], MagicMock]:
    """Factory fixture to mock httpx for Ollama API calls."""

    def _mock_client(response_data: dict[str, Any], status_code: int = 200) -> MagicMock:
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()
        if status_code >= 400:
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Error", request=MagicMock(), response=mock_response
            )

        mock_post = MagicMock(return_value=mock_response)
        mock_get = MagicMock(return_value=mock_response)
        monkeypatch.setattr(httpx, "post", mock_post)
        monkeypatch.setattr(httpx, "get", mock_get)
        return mock_post

    return _mock_client


@pytest.fixture
def mock_embedding_func(mock_embedding: list[float]) -> Generator[MagicMock, None, None]:
    """Mock get_embedding and get_embeddings_batch functions."""
    with patch("ragqa.retrieval.embeddings.get_embedding") as mock_single, patch(
        "ragqa.retrieval.embeddings.get_embeddings_batch"
    ) as mock_batch:
        mock_single.return_value = mock_embedding
        mock_batch.return_value = [mock_embedding]
        yield mock_single


@pytest.fixture
def sample_chunk_text() -> str:
    """Sample text for testing chunking."""
    return """
    ToolMem is a lifelong memory system for LLM-based agents that enables
    persistent tool usage patterns across sessions. The system uses a
    hierarchical memory structure to organize tool knowledge effectively.

    The key contributions of this paper include:
    1. A novel memory architecture for tool usage
    2. Efficient retrieval mechanisms
    3. Cross-session knowledge transfer
    """


@pytest.fixture
def sample_chunk() -> Chunk:
    """Sample chunk for testing."""
    return Chunk(
        id="test-chunk-1",
        text="This is test content about machine learning and neural networks.",
        metadata={
            "filename": "test_paper.pdf",
            "title": "Test Paper Title",
            "authors": "Smith et al.",
            "page": 1,
            "chunk_type": "content",
        },
        score=0.95,
    )


@pytest.fixture
def sample_document(sample_chunk: Chunk) -> Document:
    """Sample document for testing."""
    return Document(
        id="test-doc-1",
        filename="test_paper.pdf",
        title="Test Paper Title",
        title_variants=["Test Paper", "Paper Title"],
        authors="Smith et al.",
        abstract="This paper presents a test abstract about machine learning.",
        page_count=10,
        chunks=[sample_chunk],
    )


@pytest.fixture
def mock_streaming_response() -> Callable[[list[str]], MagicMock]:
    """Factory fixture for mocking streaming responses."""

    def _create_stream(tokens: list[str]) -> MagicMock:
        def iter_lines() -> Generator[str, None, None]:
            for token in tokens:
                yield json.dumps({"response": token, "done": False})
            yield json.dumps({"response": "", "done": True})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines = iter_lines
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        return mock_response

    return _create_stream
