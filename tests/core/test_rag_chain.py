"""Tests for RAG chain orchestration."""

from unittest.mock import MagicMock, patch

from ragqa.core.rag_chain import RAGChain, RAGResponse


class TestRAGResponse:
    """Tests for RAGResponse dataclass."""

    def test_response_creation(self) -> None:
        """RAGResponse can be created with all fields."""
        response = RAGResponse(
            answer="Test answer",
            sources=[{"filename": "test.pdf", "title": "Test"}],
            confidence=85,
            query_type="specific",
        )
        assert response.answer == "Test answer"
        assert len(response.sources) == 1
        assert response.confidence == 85
        assert response.query_type == "specific"

    def test_response_empty_sources(self) -> None:
        """RAGResponse handles empty sources."""
        response = RAGResponse(
            answer="No sources found",
            sources=[],
            confidence=0,
            query_type="specific",
        )
        assert response.sources == []
        assert response.confidence == 0

    def test_response_all_query_types(self) -> None:
        """RAGResponse supports all query types."""
        for qtype in ["all_docs", "single_doc", "specific"]:
            response = RAGResponse(
                answer="Test",
                sources=[],
                confidence=50,
                query_type=qtype,  # type: ignore[arg-type]
            )
            assert response.query_type == qtype


class TestRAGChainCleanResponse:
    """Tests for response cleaning logic."""

    def test_removes_sure_preamble(self) -> None:
        """Preambles like 'Sure!' are removed."""
        from ragqa.core.rag_chain import RAGChain

        chain = RAGChain.__new__(RAGChain)
        result = chain._clean_response("Sure! Here is the answer. The system works.")
        assert not result.startswith("Sure")

    def test_removes_based_on_preamble(self) -> None:
        """Preambles like 'Based on the context' are removed."""
        from ragqa.core.rag_chain import RAGChain

        chain = RAGChain.__new__(RAGChain)
        result = chain._clean_response(
            "Based on the provided documents. The answer is here."
        )
        assert not result.startswith("Based on")

    def test_preserves_content(self) -> None:
        """Actual content is preserved after cleaning."""
        from ragqa.core.rag_chain import RAGChain

        chain = RAGChain.__new__(RAGChain)
        result = chain._clean_response("The answer is 42.")
        assert result == "The answer is 42."


class TestRAGChainTruncation:
    """Tests for context truncation."""

    def test_truncate_respects_limit(self) -> None:
        """Truncation stops at MAX_CONTEXT_CHARS."""
        from ragqa.core.models import Chunk
        from ragqa.core.rag_chain import MAX_CONTEXT_CHARS, RAGChain

        chain = RAGChain.__new__(RAGChain)

        # Create chunks that exceed the limit
        chunks = [
            Chunk(id=f"c{i}", text="x" * 5000, metadata={}, score=0.9)
            for i in range(10)
        ]

        result = chain._truncate_context(chunks)
        total_chars = sum(len(c.text) for c in result)
        assert total_chars <= MAX_CONTEXT_CHARS

    def test_truncate_preserves_order(self) -> None:
        """Truncation preserves chunk order."""
        from ragqa.core.models import Chunk
        from ragqa.core.rag_chain import RAGChain

        chain = RAGChain.__new__(RAGChain)

        chunks = [
            Chunk(id="first", text="a" * 100, metadata={}, score=0.9),
            Chunk(id="second", text="b" * 100, metadata={}, score=0.8),
        ]

        result = chain._truncate_context(chunks)
        assert result[0].id == "first"
        assert result[1].id == "second"


class TestRAGChainSourceExtraction:
    """Tests for source extraction from chunks."""

    def test_extracts_unique_sources(self) -> None:
        """Duplicate filenames are deduplicated."""
        from ragqa.core.models import Chunk
        from ragqa.core.rag_chain import RAGChain

        chain = RAGChain.__new__(RAGChain)

        chunks = [
            Chunk(
                id="c1",
                text="text1",
                metadata={"filename": "doc1.pdf", "title": "T1", "authors": "A1"},
                score=0.9,
            ),
            Chunk(
                id="c2",
                text="text2",
                metadata={"filename": "doc1.pdf", "title": "T1", "authors": "A1"},
                score=0.8,
            ),
            Chunk(
                id="c3",
                text="text3",
                metadata={"filename": "doc2.pdf", "title": "T2", "authors": "A2"},
                score=0.7,
            ),
        ]

        sources = chain._extract_sources(chunks)
        filenames = [s["filename"] for s in sources]
        assert filenames == ["doc1.pdf", "doc2.pdf"]


class TestRAGChainRerankerWiring:
    """Tests for reranker auto-wiring in RAGChain.__init__."""

    @patch("ragqa.core.rag_chain.get_settings")
    def test_reranker_created_when_enabled(self, mock_settings: MagicMock) -> None:
        """RAGChain creates CrossEncoderReranker when reranker_enabled=True."""
        from ragqa.config import Settings

        mock_settings.return_value = Settings(
            reranker_enabled=True,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

        mock_vs = MagicMock()
        mock_bm25 = MagicMock()

        chain = RAGChain(vectorstore=mock_vs, bm25_index=mock_bm25)

        assert chain.retriever.reranker is not None

    @patch("ragqa.core.rag_chain.get_settings")
    def test_reranker_not_created_when_disabled(self, mock_settings: MagicMock) -> None:
        """RAGChain does NOT create reranker when reranker_enabled=False."""
        from ragqa.config import Settings

        mock_settings.return_value = Settings(
            reranker_enabled=False,
        )

        mock_vs = MagicMock()
        mock_bm25 = MagicMock()

        chain = RAGChain(vectorstore=mock_vs, bm25_index=mock_bm25)

        assert chain.retriever.reranker is None

    @patch("ragqa.core.rag_chain.get_settings")
    def test_explicit_reranker_overrides_settings(self, mock_settings: MagicMock) -> None:
        """Explicitly passed reranker should be used regardless of settings."""
        from ragqa.config import Settings

        mock_settings.return_value = Settings(
            reranker_enabled=False,  # disabled in settings
        )

        mock_vs = MagicMock()
        mock_bm25 = MagicMock()
        mock_reranker = MagicMock()

        chain = RAGChain(
            vectorstore=mock_vs, bm25_index=mock_bm25, reranker=mock_reranker
        )

        assert chain.retriever.reranker is mock_reranker
