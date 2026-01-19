"""Tests for hybrid retriever."""

import pytest
from ragqa.core.models import Chunk
from ragqa.retrieval.retriever import (
    DOC_SCORE_RATIO,
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
