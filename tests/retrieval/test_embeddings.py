"""Tests for embeddings module."""

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from ragqa.config import get_settings
from ragqa.exceptions import LLMError
from ragqa.llm.client import check_ollama_available
from ragqa.retrieval.embeddings import (
    check_model_available,
    get_embedding,
    get_embeddings_batch,
)


class TestGetEmbedding:
    """Tests for get_embedding function."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Clear settings cache before each test."""
        get_settings.cache_clear()

    def test_returns_vector(
        self,
        mock_httpx_client: MagicMock,
        mock_embedding_response: dict[str, Any],
    ) -> None:
        """Test that get_embedding returns a vector."""
        mock_httpx_client(mock_embedding_response)

        result = get_embedding("test text")

        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)

    def test_handles_single_embedding_format(
        self,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test handling of single embedding response format."""
        mock_httpx_client({"embedding": [0.2] * 768})

        result = get_embedding("test text")

        assert len(result) == 768

    def test_connection_error_raises_llm_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that connection errors are wrapped in LLMError."""

        def mock_post(*args: Any, **kwargs: Any) -> None:
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "post", mock_post)

        with pytest.raises(LLMError) as exc_info:
            get_embedding("test text")

        assert "Cannot connect to Ollama" in exc_info.value.message

    def test_http_error_raises_llm_error(
        self,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test that HTTP errors are wrapped in LLMError."""
        mock_httpx_client({}, status_code=500)

        with pytest.raises(LLMError) as exc_info:
            get_embedding("test text")

        assert "500" in exc_info.value.message


class TestGetEmbeddingsBatch:
    """Tests for get_embeddings_batch function."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Clear settings cache before each test."""
        get_settings.cache_clear()

    def test_returns_list_of_vectors(
        self,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test that batch returns list of vectors."""
        mock_httpx_client({"embeddings": [[0.1] * 768, [0.2] * 768]})

        result = get_embeddings_batch(["text 1", "text 2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(len(emb) == 768 for emb in result)

    def test_connection_error_raises_llm_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that connection errors are wrapped in LLMError."""

        def mock_post(*args: Any, **kwargs: Any) -> None:
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "post", mock_post)

        with pytest.raises(LLMError) as exc_info:
            get_embeddings_batch(["text 1", "text 2"])

        assert "Cannot connect to Ollama" in exc_info.value.message

    def test_http_error_raises_llm_error(
        self,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test that HTTP errors are wrapped in LLMError."""
        mock_httpx_client({}, status_code=500)

        with pytest.raises(LLMError) as exc_info:
            get_embeddings_batch(["text 1", "text 2"])

        assert "500" in exc_info.value.message


class TestCheckOllamaAvailable:
    """Tests for check_ollama_available function."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Clear settings cache before each test."""
        get_settings.cache_clear()

    def test_returns_true_when_available(
        self,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test returns True when Ollama is available."""
        mock_httpx_client({"models": []})

        result = check_ollama_available()

        assert result is True

    def test_returns_false_on_connection_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test returns False when Ollama is not available."""

        def mock_get(*args: Any, **kwargs: Any) -> None:
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "get", mock_get)

        result = check_ollama_available()

        assert result is False

    def test_returns_false_on_http_error(
        self,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test returns False on HTTP error."""
        mock_httpx_client({}, status_code=500)

        result = check_ollama_available()

        assert result is False


class TestCheckModelAvailable:
    """Tests for check_model_available function."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Clear settings cache before each test."""
        get_settings.cache_clear()

    def test_returns_true_when_model_exists(
        self,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test returns True when model is available."""
        mock_httpx_client({"models": [{"name": "qwen2.5:7b"}]})

        result = check_model_available("qwen2.5")

        assert result is True

    def test_returns_false_when_model_missing(
        self,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test returns False when model is not available."""
        mock_httpx_client({"models": [{"name": "other-model:latest"}]})

        result = check_model_available("qwen2.5")

        assert result is False

    def test_returns_false_on_connection_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test returns False when Ollama is not available."""

        def mock_get(*args: Any, **kwargs: Any) -> None:
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "get", mock_get)

        result = check_model_available("qwen2.5")

        assert result is False
