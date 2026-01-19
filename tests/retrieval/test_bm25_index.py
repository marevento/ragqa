"""Tests for BM25Index."""

from pathlib import Path

import pytest
from ragqa.config import get_settings
from ragqa.core.models import Chunk, Document
from ragqa.retrieval.bm25_index import BM25Index, tokenize


class TestTokenize:
    """Tests for tokenize function."""

    def test_basic_tokenization(self) -> None:
        """Test basic tokenization of text."""
        text = "Hello World"
        tokens = tokenize(text)
        assert tokens == ["hello", "world"]

    def test_special_chars_removed(self) -> None:
        """Test that special characters are handled properly."""
        text = "Hello, World! How are you?"
        tokens = tokenize(text)
        assert tokens == ["hello", "world", "how", "are", "you"]

    def test_numbers_preserved(self) -> None:
        """Test that numbers are preserved in tokens."""
        text = "Python 3.10 release"
        tokens = tokenize(text)
        assert "3" in tokens
        assert "10" in tokens
        assert "python" in tokens

    def test_empty_string(self) -> None:
        """Test tokenizing empty string."""
        tokens = tokenize("")
        assert tokens == []

    def test_unicode_handling(self) -> None:
        """Test handling of unicode characters."""
        text = "café résumé"
        tokens = tokenize(text)
        assert "café" in tokens or "caf" in tokens  # Depends on regex

    def test_case_normalization(self) -> None:
        """Test that text is lowercased."""
        text = "HELLO World HeLLo"
        tokens = tokenize(text)
        assert all(t.islower() or t.isdigit() for t in tokens)


class TestBM25Index:
    """Tests for BM25Index class."""

    @pytest.fixture(autouse=True)
    def setup(self, temp_dir: Path) -> None:
        """Set up test fixtures."""
        get_settings.cache_clear()
        self.persist_dir = temp_dir / "bm25"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    @pytest.fixture
    def sample_documents(self) -> list[Document]:
        """Create sample documents for testing."""
        return [
            Document(
                id="doc-1",
                filename="paper1.pdf",
                title="Machine Learning Fundamentals",
                title_variants=["ML Fundamentals"],
                authors="Smith et al.",
                abstract="A paper about machine learning basics.",
                page_count=10,
                chunks=[
                    Chunk(
                        id="chunk-1",
                        text="Machine learning is a subset of artificial intelligence.",
                        metadata={
                            "filename": "paper1.pdf",
                            "title": "Machine Learning Fundamentals",
                            "page": 1,
                        },
                    ),
                    Chunk(
                        id="chunk-2",
                        text="Deep learning uses neural networks with many layers.",
                        metadata={
                            "filename": "paper1.pdf",
                            "title": "Machine Learning Fundamentals",
                            "page": 2,
                        },
                    ),
                ],
            ),
            Document(
                id="doc-2",
                filename="paper2.pdf",
                title="Natural Language Processing",
                title_variants=["NLP"],
                authors="Jones et al.",
                abstract="A paper about NLP techniques.",
                page_count=15,
                chunks=[
                    Chunk(
                        id="chunk-3",
                        text="Natural language processing enables computers to understand text.",
                        metadata={
                            "filename": "paper2.pdf",
                            "title": "Natural Language Processing",
                            "page": 1,
                        },
                    ),
                ],
            ),
        ]

    def test_build_from_documents(self, sample_documents: list[Document]) -> None:
        """Test building index from documents."""
        bm25 = BM25Index(persist_dir=self.persist_dir)
        bm25.build_from_documents(sample_documents)

        assert bm25.is_indexed()
        assert bm25.chunk_count() == 3

    def test_search_returns_relevant_chunks(self, sample_documents: list[Document]) -> None:
        """Test that search returns relevant chunks."""
        bm25 = BM25Index(persist_dir=self.persist_dir)
        bm25.build_from_documents(sample_documents)

        results = bm25.search("machine learning", top_k=5)

        assert len(results) >= 1
        # Should return chunks mentioning machine learning
        assert any("machine" in r.text.lower() for r in results)

    def test_search_normalizes_scores(self, sample_documents: list[Document]) -> None:
        """Test that search scores are normalized to 0-1 range."""
        bm25 = BM25Index(persist_dir=self.persist_dir)
        bm25.build_from_documents(sample_documents)

        results = bm25.search("neural networks", top_k=5)

        assert all(0 <= r.score <= 1 for r in results)

    def test_clear_removes_index(self, sample_documents: list[Document]) -> None:
        """Test that clear removes the index."""
        bm25 = BM25Index(persist_dir=self.persist_dir)
        bm25.build_from_documents(sample_documents)

        assert bm25.is_indexed()

        bm25.clear()

        assert not bm25.is_indexed()
        assert bm25.chunk_count() == 0

    def test_persistence_save_load(self, sample_documents: list[Document]) -> None:
        """Test that index persists across instances."""
        bm25 = BM25Index(persist_dir=self.persist_dir)
        bm25.build_from_documents(sample_documents)

        # Create new instance pointing to same directory
        bm25_loaded = BM25Index(persist_dir=self.persist_dir)

        assert bm25_loaded.is_indexed()
        assert bm25_loaded.chunk_count() == bm25.chunk_count()

        # Search should work on loaded index
        results = bm25_loaded.search("machine learning", top_k=5)
        assert len(results) >= 1

    def test_add_document_incremental(self, sample_documents: list[Document]) -> None:
        """Test adding documents incrementally."""
        bm25 = BM25Index(persist_dir=self.persist_dir)
        bm25.build_from_documents([sample_documents[0]])

        initial_count = bm25.chunk_count()

        bm25.add_document(sample_documents[1])

        assert bm25.chunk_count() == initial_count + len(sample_documents[1].chunks)

    def test_search_empty_index(self) -> None:
        """Test searching empty index returns empty list."""
        bm25 = BM25Index(persist_dir=self.persist_dir)

        results = bm25.search("test query", top_k=5)

        assert results == []

    def test_search_no_matches(self, sample_documents: list[Document]) -> None:
        """Test search with no matching terms."""
        bm25 = BM25Index(persist_dir=self.persist_dir)
        bm25.build_from_documents(sample_documents)

        results = bm25.search("xyzabc123nonexistent", top_k=5)

        # Should return empty or very low score results
        assert len(results) == 0 or all(r.score == 0 for r in results)

    def test_title_tokens_included(self, sample_documents: list[Document]) -> None:
        """Test that title tokens are included in corpus for better matching."""
        bm25 = BM25Index(persist_dir=self.persist_dir)
        bm25.build_from_documents(sample_documents)

        # Search for title term should return relevant chunks
        results = bm25.search("Natural Language Processing", top_k=5)

        # Should find chunks from that paper
        assert any(r.metadata.get("filename") == "paper2.pdf" for r in results)

    def test_top_k_limits_results(self, sample_documents: list[Document]) -> None:
        """Test that top_k parameter limits results."""
        bm25 = BM25Index(persist_dir=self.persist_dir)
        bm25.build_from_documents(sample_documents)

        results = bm25.search("paper", top_k=2)

        assert len(results) <= 2
