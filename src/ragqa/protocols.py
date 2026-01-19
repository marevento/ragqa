"""Protocol definitions for type-safe dependency injection.

These protocols define the interfaces for the core components of the RAG system,
enabling dependency injection and testability.
"""

from collections.abc import AsyncGenerator, Generator
from typing import Any, Protocol

from ragqa.core.models import Chunk, Document


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation."""

    def get_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of embedding values.
        """
        ...

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...


class AsyncEmbeddingProvider(Protocol):
    """Protocol for async embedding generation."""

    async def get_embedding_async(self, text: str) -> list[float]:
        """Generate embedding for a single text (async).

        Args:
            text: The text to embed.

        Returns:
            List of embedding values.
        """
        ...

    async def get_embeddings_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts (async).

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...


class VectorStoreProtocol(Protocol):
    """Protocol for vector store operations."""

    def add_document(self, document: Document) -> None:
        """Add a document to the vector store.

        Args:
            document: The document to add.
        """
        ...

    def search_chunks(self, query: str, top_k: int) -> list[Chunk]:
        """Search for similar chunks.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of matching chunks with scores.
        """
        ...

    def search_documents(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Search for similar documents.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of matching documents with scores.
        """
        ...

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Get all indexed documents.

        Returns:
            List of all documents.
        """
        ...

    def get_title_embeddings(self) -> dict[str, list[float]]:
        """Get cached title embeddings.

        Returns:
            Dict mapping filename to title embedding.
        """
        ...

    def document_count(self) -> int:
        """Get number of indexed documents."""
        ...

    def chunk_count(self) -> int:
        """Get number of indexed chunks."""
        ...

    def clear(self) -> None:
        """Clear all collections."""
        ...

    def is_indexed(self) -> bool:
        """Check if any documents have been indexed."""
        ...


class AsyncVectorStoreProtocol(Protocol):
    """Protocol for async vector store operations."""

    async def search_chunks_async(self, query: str, top_k: int) -> list[Chunk]:
        """Search for similar chunks (async).

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of matching chunks with scores.
        """
        ...

    async def search_documents_async(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Search for similar documents (async).

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of matching documents with scores.
        """
        ...


class KeywordIndex(Protocol):
    """Protocol for keyword-based search index."""

    def build_from_documents(self, documents: list[Document]) -> None:
        """Build index from documents.

        Args:
            documents: List of documents to index.
        """
        ...

    def add_document(self, document: Document) -> None:
        """Add a single document to the index.

        Args:
            document: The document to add.
        """
        ...

    def search(self, query: str, top_k: int) -> list[Chunk]:
        """Search for chunks matching the query.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of matching chunks with scores.
        """
        ...

    def clear(self) -> None:
        """Clear the index."""
        ...

    def is_indexed(self) -> bool:
        """Check if index has been built."""
        ...

    def chunk_count(self) -> int:
        """Get number of indexed chunks."""
        ...


class LLMProvider(Protocol):
    """Protocol for LLM generation."""

    def generate(self, prompt: str, stream: bool = False) -> str | Generator[str, None, None]:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to send to the LLM.
            stream: Whether to stream the response.

        Returns:
            If stream=False, returns complete response string.
            If stream=True, returns generator yielding tokens.
        """
        ...


class AsyncLLMProvider(Protocol):
    """Protocol for async LLM generation."""

    async def generate_async(
        self, prompt: str, stream: bool = False
    ) -> str | AsyncGenerator[str, None]:
        """Generate text from a prompt (async).

        Args:
            prompt: The prompt to send to the LLM.
            stream: Whether to stream the response.

        Returns:
            If stream=False, returns complete response string.
            If stream=True, returns async generator yielding tokens.
        """
        ...


class RetrieverProtocol(Protocol):
    """Protocol for hybrid retrieval."""

    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Retrieve relevant chunks.

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of relevant chunks with scores.
        """
        ...

    def calculate_confidence(self, chunks: list[Chunk]) -> int:
        """Calculate confidence score from chunks.

        Args:
            chunks: Retrieved chunks.

        Returns:
            Confidence score 0-100.
        """
        ...


class AsyncRetrieverProtocol(Protocol):
    """Protocol for async hybrid retrieval."""

    async def retrieve_async(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Retrieve relevant chunks (async).

        Args:
            query: The search query.
            top_k: Maximum number of results.

        Returns:
            List of relevant chunks with scores.
        """
        ...
