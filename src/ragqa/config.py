"""Configuration management using pydantic-settings."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"

    # Model parameters
    llm_temperature: float = 0.1
    llm_context_window: int = 8192

    # Papers directory
    papers_dir: Path = Path("./research_papers")

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Retrieval settings
    retrieval_top_k: int = 5

    # ChromaDB settings
    chroma_persist_dir: Path = Path("./.chroma_db")

    # Logging
    log_level: str = "WARNING"
    debug: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()
