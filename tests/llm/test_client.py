"""Tests for LLM client module."""

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from ragqa.config import get_settings
from ragqa.exceptions import LLMError
from ragqa.llm.client import (
    check_ollama_available,
    generate,
    get_available_models,
)


class TestGenerate:
    """Tests for generate function."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Clear settings cache before each test."""
        get_settings.cache_clear()

    def test_sync_returns_string(
        self,
        mock_httpx_client: MagicMock,
        mock_generate_response: dict[str, Any],
    ) -> None:
        """Test that sync generation returns a string."""
        mock_httpx_client(mock_generate_response)

        result = generate("Test prompt", stream=False)

        assert isinstance(result, str)
        assert "Test answer" in result

    def test_stream_yields_tokens(
        self,
        mock_streaming_response: MagicMock,
    ) -> None:
        """Test that streaming yields tokens."""
        mock_response = mock_streaming_response(["Hello", " ", "World"])

        with patch("httpx.stream") as mock_stream:
            mock_stream.return_value = mock_response

            result = generate("Test prompt", stream=True)
            tokens = list(result)

        assert len(tokens) >= 1
        assert "Hello" in tokens

    def test_connection_error_handling_sync(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test connection error handling in sync mode."""

        def mock_post(*args: Any, **kwargs: Any) -> None:
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "post", mock_post)

        with pytest.raises(LLMError) as exc_info:
            generate("Test prompt", stream=False)

        assert "Cannot connect to Ollama" in exc_info.value.message

    def test_connection_error_handling_stream(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test connection error handling in stream mode."""

        def mock_stream(*args: Any, **kwargs: Any) -> None:
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "stream", mock_stream)

        with pytest.raises(LLMError) as exc_info:
            result = generate("Test prompt", stream=True)
            # Need to iterate to trigger the generator
            list(result)

        assert "Cannot connect to Ollama" in exc_info.value.message

    def test_http_error_handling_sync(
        self,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test HTTP error handling in sync mode."""
        mock_httpx_client({}, status_code=500)

        with pytest.raises(LLMError) as exc_info:
            generate("Test prompt", stream=False)

        assert "500" in exc_info.value.message

    def test_empty_response_handling(
        self,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test handling of empty response."""
        mock_httpx_client({"response": "", "done": True})

        result = generate("Test prompt", stream=False)

        assert result == ""


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


class TestGetAvailableModels:
    """Tests for get_available_models function."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Clear settings cache before each test."""
        get_settings.cache_clear()

    def test_returns_model_list(
        self,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test returns list of available models."""
        mock_httpx_client(
            {"models": [{"name": "qwen2.5:7b"}, {"name": "nomic-embed-text:latest"}]}
        )

        result = get_available_models()

        assert isinstance(result, list)
        assert "qwen2.5:7b" in result
        assert "nomic-embed-text:latest" in result

    def test_returns_empty_on_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test returns empty list on error."""

        def mock_get(*args: Any, **kwargs: Any) -> None:
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx, "get", mock_get)

        result = get_available_models()

        assert result == []

    def test_returns_empty_on_http_error(
        self,
        mock_httpx_client: MagicMock,
    ) -> None:
        """Test returns empty list on HTTP error."""
        mock_httpx_client({}, status_code=500)

        result = get_available_models()

        assert result == []
