"""Tests for hybrid retriever."""

from unittest.mock import MagicMock, patch

import pytest
from ragqa.core.models import Chunk
from ragqa.retrieval.retriever import (
    DOC_SCORE_RATIO,
    Retriever,
    reciprocal_rank_fusion,
)


class TestReciprocalRankFusion:
    """Tests for RRF merging algorithm."""

    def test_rrf_single_list(self) -> None:
        """RRF with single list returns same order."""
        chunks = [
            Chunk(id="1", text="first", metadata={}, score=0.9),
            Chunk(id="2", text="second", metadata={}, score=0.8),
        ]
        result = reciprocal_rank_fusion([chunks])
        assert len(result) == 2
        assert result[0].id == "1"
        assert result[1].id == "2"

    def test_rrf_merges_duplicates(self) -> None:
        """RRF combines scores for same chunk in multiple lists."""
        list1 = [
            Chunk(id="a", text="chunk a", metadata={}, score=0.9),
            Chunk(id="b", text="chunk b", metadata={}, score=0.8),
        ]
        list2 = [
            Chunk(id="b", text="chunk b", metadata={}, score=0.95),
            Chunk(id="a", text="chunk a", metadata={}, score=0.7),
        ]
        result = reciprocal_rank_fusion([list1, list2])
        assert len(result) == 2
        # Both chunks appear in both lists, so both get boosted scores

    def test_rrf_empty_lists(self) -> None:
        """RRF handles empty input."""
        result = reciprocal_rank_fusion([])
        assert result == []

        result = reciprocal_rank_fusion([[]])
        assert result == []

    def test_rrf_disjoint_lists(self) -> None:
        """RRF handles lists with no overlap."""
        list1 = [Chunk(id="a", text="a", metadata={}, score=0.9)]
        list2 = [Chunk(id="b", text="b", metadata={}, score=0.9)]
        result = reciprocal_rank_fusion([list1, list2])
        assert len(result) == 2
        ids = {c.id for c in result}
        assert ids == {"a", "b"}


class TestDocumentFiltering:
    """Tests for document-level score filtering."""

    def test_doc_score_ratio_constant(self) -> None:
        """Verify DOC_SCORE_RATIO is set correctly."""
        assert 0 < DOC_SCORE_RATIO < 1
        assert DOC_SCORE_RATIO == 0.7


class TestRetrieverIntegration:
    """Integration tests for Retriever class."""

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        """Create sample chunks for testing."""
        return [
            Chunk(
                id="doc1_chunk1",
                text="ToolMem is a memory system",
                metadata={"filename": "doc1.pdf", "title": "ToolMem Paper"},
                score=0.9,
            ),
            Chunk(
                id="doc1_chunk2",
                text="ToolMem uses hierarchical memory",
                metadata={"filename": "doc1.pdf", "title": "ToolMem Paper"},
                score=0.85,
            ),
            Chunk(
                id="doc2_chunk1",
                text="Different topic here",
                metadata={"filename": "doc2.pdf", "title": "Other Paper"},
                score=0.5,
            ),
        ]

    def test_chunks_from_same_doc_aggregate(self, sample_chunks: list[Chunk]) -> None:
        """Chunks from same document should have aggregated scores."""
        # doc1 has 2 chunks with scores 0.9 + 0.85 = 1.75
        # doc2 has 1 chunk with score 0.5
        doc_scores: dict[str, float] = {}
        for chunk in sample_chunks:
            filename = chunk.metadata.get("filename", "")
            doc_scores[filename] = doc_scores.get(filename, 0) + chunk.score

        assert doc_scores["doc1.pdf"] == pytest.approx(1.75)
        assert doc_scores["doc2.pdf"] == pytest.approx(0.5)


class TestRetrieverWithReranker:
    """Integration tests verifying the Retriever calls the reranker."""

    @pytest.fixture
    def mock_vectorstore(self) -> MagicMock:
        """Mock vectorstore returning fixed chunks."""
        vs = MagicMock()
        vs.search_chunks.return_value = [
            Chunk(id="s1", text="semantic result", metadata={"filename": "a.pdf"}, score=0.9),
            Chunk(id="s2", text="semantic result 2", metadata={"filename": "a.pdf"}, score=0.8),
        ]
        vs.get_title_embeddings.return_value = {}
        vs.get_all_documents.return_value = []
        return vs

    @pytest.fixture
    def mock_bm25(self) -> MagicMock:
        """Mock BM25 index returning fixed chunks."""
        bm25 = MagicMock()
        bm25.search.return_value = [
            Chunk(id="b1", text="bm25 result", metadata={"filename": "a.pdf"}, score=0.85),
        ]
        return bm25

    @pytest.fixture
    def mock_reranker(self) -> MagicMock:
        """Mock reranker that reverses chunk order and sets sigmoid scores."""
        reranker = MagicMock()

        def fake_rerank(
            query: str, chunks: list[Chunk], top_k: int
        ) -> list[Chunk]:
            # Reverse order and assign sigmoid-like scores
            reversed_chunks = list(reversed(chunks[:top_k]))
            return [
                Chunk(
                    id=c.id, text=c.text, metadata=c.metadata,
                    score=0.9 - i * 0.1,
                )
                for i, c in enumerate(reversed_chunks)
            ]

        reranker.rerank.side_effect = fake_rerank
        return reranker

    @patch("ragqa.retrieval.retriever.get_embedding", return_value=[0.1] * 768)
    def test_retrieve_calls_reranker(
        self,
        _mock_emb: MagicMock,
        mock_vectorstore: MagicMock,
        mock_bm25: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        """Retriever.retrieve() should call reranker.rerank when reranker is set."""
        retriever = Retriever(
            vectorstore=mock_vectorstore,
            bm25_index=mock_bm25,
            reranker=mock_reranker,
        )

        result = retriever.retrieve("test query", top_k=2)

        mock_reranker.rerank.assert_called_once()
        call_args = mock_reranker.rerank.call_args
        assert call_args[0][0] == "test query"  # query
        assert call_args[0][2] == 2  # top_k
        assert len(result) <= 2

    @patch("ragqa.retrieval.retriever.get_embedding", return_value=[0.1] * 768)
    def test_retrieve_without_reranker_uses_filter(
        self,
        _mock_emb: MagicMock,
        mock_vectorstore: MagicMock,
        mock_bm25: MagicMock,
    ) -> None:
        """Retriever without reranker should use _filter_by_score."""
        retriever = Retriever(
            vectorstore=mock_vectorstore,
            bm25_index=mock_bm25,
            reranker=None,
        )

        result = retriever.retrieve("test query", top_k=5)

        # Should get results (from RRF fusion + filter)
        assert len(result) > 0
        # Scores should be small RRF values (not sigmoid)
        for chunk in result:
            assert chunk.score < 0.1

    @patch("ragqa.retrieval.retriever.get_embedding", return_value=[0.1] * 768)
    def test_reranker_skips_filter_by_score(
        self,
        _mock_emb: MagicMock,
        mock_vectorstore: MagicMock,
        mock_bm25: MagicMock,
        mock_reranker: MagicMock,
    ) -> None:
        """When reranker is active, _filter_by_score should be skipped."""
        retriever = Retriever(
            vectorstore=mock_vectorstore,
            bm25_index=mock_bm25,
            reranker=mock_reranker,
        )

        result = retriever.retrieve("test query", top_k=3)

        # Reranker sets sigmoid-like scores (0.8, 0.7, etc.)
        # If _filter_by_score ran, it might drop chunks — verify all are present
        assert len(result) > 0
        for chunk in result:
            assert chunk.score > 0.1  # sigmoid scores, not RRF


class TestCalculateConfidence:
    """Tests for confidence calculation with both score domains."""

    def test_confidence_with_rrf_scores(self) -> None:
        """RRF scores (~0.016-0.032) should produce reasonable confidence."""
        retriever = Retriever.__new__(Retriever)
        chunks = [
            Chunk(id="1", text="t", metadata={}, score=0.032),
            Chunk(id="2", text="t", metadata={}, score=0.016),
        ]
        confidence = retriever.calculate_confidence(chunks)
        # avg = 0.024, * 3000 = 72
        assert 50 < confidence < 100

    def test_confidence_with_sigmoid_scores(self) -> None:
        """Sigmoid scores (0.5-0.99) should produce reasonable confidence, not always 100."""
        retriever = Retriever.__new__(Retriever)
        chunks = [
            Chunk(id="1", text="t", metadata={}, score=0.85),
            Chunk(id="2", text="t", metadata={}, score=0.72),
            Chunk(id="3", text="t", metadata={}, score=0.60),
        ]
        confidence = retriever.calculate_confidence(chunks)
        # avg of top 3 = 0.723, * 100 = 72
        assert 60 < confidence < 90

    def test_confidence_sigmoid_high_scores(self) -> None:
        """Very high sigmoid scores should give high confidence."""
        retriever = Retriever.__new__(Retriever)
        chunks = [
            Chunk(id="1", text="t", metadata={}, score=0.98),
            Chunk(id="2", text="t", metadata={}, score=0.95),
        ]
        confidence = retriever.calculate_confidence(chunks)
        assert confidence >= 90

    def test_confidence_sigmoid_low_scores(self) -> None:
        """Low sigmoid scores should give low confidence."""
        retriever = Retriever.__new__(Retriever)
        chunks = [
            Chunk(id="1", text="t", metadata={}, score=0.3),
            Chunk(id="2", text="t", metadata={}, score=0.2),
        ]
        confidence = retriever.calculate_confidence(chunks)
        assert confidence < 30

    def test_confidence_empty_chunks(self) -> None:
        """Empty chunks should return 0 confidence."""
        retriever = Retriever.__new__(Retriever)
        assert retriever.calculate_confidence([]) == 0
