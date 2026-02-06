"""Tests for cross-encoder re-ranking."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ragqa.core.models import Chunk
from ragqa.retrieval.reranker import CrossEncoderReranker


def _make_chunk(id: str, text: str, score: float, **meta: str | int) -> Chunk:
    """Helper to create a chunk with metadata."""
    return Chunk(id=id, text=text, metadata=dict(meta), score=score)


@pytest.fixture
def mock_cross_encoder() -> MagicMock:
    """Mock CrossEncoder that returns predictable scores."""
    return MagicMock()


@pytest.fixture
def reranker(mock_cross_encoder: MagicMock) -> CrossEncoderReranker:
    """Create a reranker with a mocked model via the public factory."""
    return CrossEncoderReranker.from_pretrained(mock_cross_encoder)


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    def test_reranks_by_cross_encoder_score(
        self, reranker: CrossEncoderReranker, mock_cross_encoder: MagicMock
    ) -> None:
        """Reranker should re-order chunks by cross-encoder score."""
        chunks = [
            _make_chunk("a", "low relevance text", 0.9, filename="doc1.pdf"),
            _make_chunk("b", "high relevance text", 0.5, filename="doc2.pdf"),
            _make_chunk("c", "medium relevance text", 0.7, filename="doc3.pdf"),
        ]
        # Cross-encoder scores: b > c > a (different from original order)
        mock_cross_encoder.predict.return_value = np.array([-2.0, 3.0, 1.0])

        result = reranker.rerank("test query", chunks, top_k=3)

        assert len(result) == 3
        assert result[0].id == "b"  # highest cross-encoder score
        assert result[1].id == "c"
        assert result[2].id == "a"  # lowest cross-encoder score

    def test_scores_normalized_via_sigmoid(
        self, reranker: CrossEncoderReranker, mock_cross_encoder: MagicMock
    ) -> None:
        """Output scores should be normalized to 0-1 via sigmoid."""
        chunks = [_make_chunk("a", "text", 0.5)]
        mock_cross_encoder.predict.return_value = np.array([0.0])

        result = reranker.rerank("query", chunks, top_k=1)

        # sigmoid(0) = 0.5
        assert result[0].score == pytest.approx(0.5)

    def test_high_score_near_one(
        self, reranker: CrossEncoderReranker, mock_cross_encoder: MagicMock
    ) -> None:
        """High raw scores should produce scores near 1.0."""
        chunks = [_make_chunk("a", "text", 0.5)]
        mock_cross_encoder.predict.return_value = np.array([10.0])

        result = reranker.rerank("query", chunks, top_k=1)

        assert result[0].score > 0.99

    def test_empty_chunks_returns_empty(self, reranker: CrossEncoderReranker) -> None:
        """Reranking empty input returns empty list without calling model."""
        result = reranker.rerank("query", [], top_k=5)

        assert result == []

    def test_top_k_truncation(
        self, reranker: CrossEncoderReranker, mock_cross_encoder: MagicMock
    ) -> None:
        """Result should be truncated to top_k."""
        chunks = [
            _make_chunk("a", "text a", 0.9),
            _make_chunk("b", "text b", 0.8),
            _make_chunk("c", "text c", 0.7),
            _make_chunk("d", "text d", 0.6),
        ]
        mock_cross_encoder.predict.return_value = np.array([1.0, 4.0, 3.0, 2.0])

        result = reranker.rerank("query", chunks, top_k=2)

        assert len(result) == 2
        assert result[0].id == "b"  # score 4.0 -> highest
        assert result[1].id == "c"  # score 3.0 -> second

    def test_metadata_preserved(
        self, reranker: CrossEncoderReranker, mock_cross_encoder: MagicMock
    ) -> None:
        """Original chunk metadata should be preserved through reranking."""
        metadata = {
            "filename": "paper.pdf",
            "title": "Important Paper",
            "authors": "Smith et al.",
            "page": 5,
        }
        chunks = [Chunk(id="x", text="some text", metadata=metadata, score=0.9)]
        mock_cross_encoder.predict.return_value = np.array([2.0])

        result = reranker.rerank("query", chunks, top_k=1)

        assert result[0].id == "x"
        assert result[0].text == "some text"
        assert result[0].metadata == metadata
        assert result[0].filename == "paper.pdf"
        assert result[0].title == "Important Paper"
        assert result[0].authors == "Smith et al."
        assert result[0].page == 5

    def test_lazy_model_loading(self) -> None:
        """Construction should not load the model."""
        reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        # Model is None immediately after construction (not loaded yet)
        assert reranker._model is None
        assert not reranker._load_failed

    def test_model_loaded_on_first_call(self) -> None:
        """Model should be loaded on first rerank call."""
        reranker = CrossEncoderReranker()
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0])

        with patch(
            "sentence_transformers.CrossEncoder", return_value=mock_model
        ):
            chunks = [_make_chunk("a", "text", 0.5)]
            reranker.rerank("query", chunks, top_k=1)

        assert reranker._model is mock_model

    def test_model_reused_across_calls(
        self, reranker: CrossEncoderReranker, mock_cross_encoder: MagicMock
    ) -> None:
        """Model should be reused across multiple rerank calls."""
        chunks = [_make_chunk("a", "text", 0.5)]
        mock_cross_encoder.predict.return_value = np.array([1.0])

        reranker.rerank("query1", chunks, top_k=1)
        reranker.rerank("query2", chunks, top_k=1)

        assert mock_cross_encoder.predict.call_count == 2

    def test_pairs_passed_to_model(
        self, reranker: CrossEncoderReranker, mock_cross_encoder: MagicMock
    ) -> None:
        """Model.predict should receive (query, passage) pairs."""
        chunks = [
            _make_chunk("a", "first passage", 0.9),
            _make_chunk("b", "second passage", 0.8),
        ]
        mock_cross_encoder.predict.return_value = np.array([1.0, 2.0])

        reranker.rerank("my query", chunks, top_k=2)

        expected_pairs = [["my query", "first passage"], ["my query", "second passage"]]
        mock_cross_encoder.predict.assert_called_once_with(expected_pairs)


class TestCrossEncoderErrorHandling:
    """Tests for graceful degradation when model fails to load."""

    def test_load_failure_returns_chunks_unchanged(self) -> None:
        """If model fails to load, rerank returns input truncated to top_k."""
        reranker = CrossEncoderReranker(model_name="nonexistent/model")

        with patch(
            "sentence_transformers.CrossEncoder",
            side_effect=OSError("Download failed"),
        ):
            chunks = [
                _make_chunk("a", "text a", 0.9),
                _make_chunk("b", "text b", 0.8),
                _make_chunk("c", "text c", 0.7),
            ]
            result = reranker.rerank("query", chunks, top_k=2)

        # Should return first top_k chunks unchanged (graceful fallback)
        assert len(result) == 2
        assert result[0].id == "a"
        assert result[1].id == "b"
        # Scores unchanged (not sigmoid-transformed)
        assert result[0].score == pytest.approx(0.9)

    def test_load_failure_is_sticky(self) -> None:
        """After load failure, subsequent calls don't retry."""
        reranker = CrossEncoderReranker(model_name="bad/model")

        with patch(
            "sentence_transformers.CrossEncoder",
            side_effect=OSError("Network error"),
        ) as mock_ctor:
            chunks = [_make_chunk("a", "text", 0.5)]
            reranker.rerank("query1", chunks, top_k=1)
            reranker.rerank("query2", chunks, top_k=1)

            # Constructor should only be called once (failure is cached)
            assert mock_ctor.call_count == 1

    def test_from_pretrained_skips_loading(self) -> None:
        """from_pretrained should not trigger lazy loading."""
        mock_model = MagicMock()
        reranker = CrossEncoderReranker.from_pretrained(mock_model)

        assert reranker._model is mock_model
        assert not reranker._load_failed
