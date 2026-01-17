"""Shared test fixtures."""

from pathlib import Path

import pytest
from ragqa.config import Settings


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def mock_settings(temp_dir: Path) -> Settings:
    """Provide mock settings for testing."""
    return Settings(
        ollama_base_url="http://localhost:11434",
        ollama_model="test-model",
        ollama_embed_model="test-embed",
        papers_dir=temp_dir / "papers",
        chroma_persist_dir=temp_dir / "chroma",
    )


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
