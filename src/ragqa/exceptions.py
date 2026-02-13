"""Custom exception hierarchy for RAG Q&A."""


class RAGError(Exception):
    """Base exception for RAG Q&A."""

    def __init__(self, message: str, details: str | None = None) -> None:
        self.message = message
        self.details = details
        super().__init__(message)


class PDFError(RAGError):
    """PDF parsing/extraction errors."""

    pass


class IndexingError(RAGError):
    """Vector/BM25 index errors."""

    pass


class LLMError(RAGError):
    """Ollama/LLM errors."""

    pass


class ConfigError(RAGError):
    """Configuration/environment errors."""

    pass
