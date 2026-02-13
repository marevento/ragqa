"""Tests for VectorStore."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ragqa.config import get_settings
from ragqa.core.models import Chunk, Document
from ragqa.exceptions import IndexingError as RAGIndexError
from ragqa.retrieval.vectorstore import VectorStore


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.fixture(autouse=True)
    def setup(self, temp_dir: Path) -> None:
        """Set up test fixtures."""
        get_settings.cache_clear()
        self.persist_dir = temp_dir / "chroma"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    @pytest.fixture
    def mock_embeddings(self, mock_embedding: list[float]) -> MagicMock:
        """Mock embedding functions."""
        with patch("ragqa.retrieval.vectorstore.get_embedding") as mock_single, patch(
            "ragqa.retrieval.vectorstore.get_embeddings_batch"
        ) as mock_batch:
            mock_single.return_value = mock_embedding
            mock_batch.return_value = [mock_embedding]
            yield mock_single

    def test_add_document_creates_chunks(
        self,
        sample_document: Document,
        mock_embeddings: MagicMock,
        mock_embedding: list[float],
    ) -> None:
        """Test that adding a document creates entries in both collections."""
        vectorstore = VectorStore(persist_dir=self.persist_dir)
        vectorstore.add_document(sample_document)

        assert vectorstore.document_count() == 1
        assert vectorstore.chunk_count() == len(sample_document.chunks)

    def test_search_chunks_returns_scored_results(
        self,
        sample_document: Document,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test that search returns chunks with similarity scores."""
        vectorstore = VectorStore(persist_dir=self.persist_dir)
        vectorstore.add_document(sample_document)

        results = vectorstore.search_chunks("machine learning", top_k=5)

        assert len(results) >= 1
        assert all(isinstance(chunk, Chunk) for chunk in results)
        assert all(hasattr(chunk, "score") for chunk in results)
        # Scores should be between 0 and 1 (cosine similarity)
        assert all(0 <= chunk.score <= 1 for chunk in results)

    def test_search_empty_index_raises_error(
        self,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test that searching an empty index raises appropriate error."""
        vectorstore = VectorStore(persist_dir=self.persist_dir)

        with pytest.raises(RAGIndexError) as exc_info:
            vectorstore.search_chunks("test query")

        assert "No documents indexed" in str(exc_info.value.message)

    def test_document_count_accuracy(
        self,
        sample_document: Document,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test that document count accurately reflects indexed documents."""
        vectorstore = VectorStore(persist_dir=self.persist_dir)

        assert vectorstore.document_count() == 0

        vectorstore.add_document(sample_document)
        assert vectorstore.document_count() == 1

        # Add another document
        doc2 = Document(
            id="test-doc-2",
            filename="test_paper_2.pdf",
            title="Another Test Paper",
            title_variants=[],
            authors="Jones et al.",
            abstract="Another abstract.",
            page_count=5,
            chunks=[
                Chunk(
                    id="test-chunk-2",
                    text="Another chunk text.",
                    metadata={"filename": "test_paper_2.pdf", "title": "Another Test Paper"},
                )
            ],
        )
        vectorstore.add_document(doc2)
        assert vectorstore.document_count() == 2

    def test_clear_removes_all_data(
        self,
        sample_document: Document,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test that clear removes all documents and chunks."""
        vectorstore = VectorStore(persist_dir=self.persist_dir)
        vectorstore.add_document(sample_document)

        assert vectorstore.document_count() > 0
        assert vectorstore.chunk_count() > 0

        vectorstore.clear()

        assert vectorstore.document_count() == 0
        assert vectorstore.chunk_count() == 0

    def test_is_indexed_state(
        self,
        sample_document: Document,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test is_indexed returns correct state."""
        vectorstore = VectorStore(persist_dir=self.persist_dir)

        assert not vectorstore.is_indexed()

        vectorstore.add_document(sample_document)
        assert vectorstore.is_indexed()

        vectorstore.clear()
        assert not vectorstore.is_indexed()

    def test_get_all_documents(
        self,
        sample_document: Document,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test retrieving all indexed documents."""
        vectorstore = VectorStore(persist_dir=self.persist_dir)
        vectorstore.add_document(sample_document)

        docs = vectorstore.get_all_documents()

        assert len(docs) == 1
        assert docs[0]["metadata"]["filename"] == sample_document.filename
        assert docs[0]["metadata"]["title"] == sample_document.title

    def test_search_documents(
        self,
        sample_document: Document,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test document-level search."""
        vectorstore = VectorStore(persist_dir=self.persist_dir)
        vectorstore.add_document(sample_document)

        docs = vectorstore.search_documents("machine learning", top_k=5)

        assert len(docs) >= 1
        assert all("score" in doc for doc in docs)
        assert all(0 <= doc["score"] <= 1 for doc in docs)

    def test_title_embeddings_stored_and_retrieved(
        self,
        sample_document: Document,
        mock_embeddings: MagicMock,
        mock_embedding: list[float],
    ) -> None:
        """Test that title embeddings are stored and can be retrieved."""
        vectorstore = VectorStore(persist_dir=self.persist_dir)
        vectorstore.add_document(sample_document)

        title_embeddings = vectorstore.get_title_embeddings()

        assert sample_document.filename in title_embeddings
        # Should have same dimension as mock embedding
        assert len(title_embeddings[sample_document.filename]) == len(mock_embedding)

    def test_upsert_updates_existing_document(
        self,
        sample_document: Document,
        mock_embeddings: MagicMock,
    ) -> None:
        """Test that adding same document ID updates rather than duplicates."""
        vectorstore = VectorStore(persist_dir=self.persist_dir)
        vectorstore.add_document(sample_document)

        # Add same document again (should upsert)
        vectorstore.add_document(sample_document)

        assert vectorstore.document_count() == 1
