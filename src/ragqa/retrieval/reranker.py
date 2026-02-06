"""Cross-encoder re-ranking for improved retrieval precision."""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from ragqa import get_logger
from ragqa.core.models import Chunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = get_logger(__name__)


class CrossEncoderReranker:
    """Re-ranks chunks using a cross-encoder model.

    The cross-encoder scores (query, passage) pairs jointly, producing
    more accurate relevance scores than bi-encoder similarity or BM25.
    The model is lazy-loaded on first call to avoid startup cost.

    If the model fails to load (network error, missing model, etc.),
    rerank() returns the input chunks unchanged rather than crashing
    the pipeline.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None
        self._load_failed: bool = False

    @classmethod
    def from_pretrained(cls, model: CrossEncoder) -> CrossEncoderReranker:
        """Create a reranker with a pre-loaded model (useful for testing)."""
        instance = cls()
        instance._model = model
        return instance

    def _get_model(self) -> CrossEncoder | None:
        """Lazy-load the cross-encoder model.

        Returns None if loading fails, so callers can gracefully degrade.
        """
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None

        try:
            from sentence_transformers import CrossEncoder

            logger.info("loading_cross_encoder", model=self._model_name)
            self._model = CrossEncoder(self._model_name)
        except Exception:
            logger.warning(
                "cross_encoder_load_failed",
                model=self._model_name,
                exc_info=True,
            )
            self._load_failed = True
            return None
        return self._model

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        """Re-rank chunks by cross-encoder relevance score.

        Args:
            query: The search query.
            chunks: Candidate chunks from first-stage retrieval.
            top_k: Maximum number of results to return.

        Returns:
            Top-k chunks sorted by cross-encoder score, with scores
            normalized to 0-1 via sigmoid. Falls back to returning
            the input chunks (truncated to top_k) if the model is
            unavailable.
        """
        if not chunks:
            return []

        start = time.time()
        model = self._get_model()

        # Graceful degradation: if model failed to load, pass through
        if model is None:
            return chunks[:top_k]

        # Score each (query, passage) pair
        pairs = [[query, chunk.text] for chunk in chunks]
        raw_scores: Any = model.predict(pairs)

        scored: list[tuple[float, Chunk]] = []
        for chunk, raw in zip(chunks, raw_scores, strict=True):
            normalized = 1.0 / (1.0 + math.exp(-float(raw)))
            scored.append((normalized, chunk))

        # Sort by cross-encoder score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build result with updated scores, preserving all metadata
        result: list[Chunk] = []
        for score, chunk in scored[:top_k]:
            result.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=score,
                )
            )

        duration_ms = (time.time() - start) * 1000
        logger.info(
            "reranking_completed",
            duration_ms=round(duration_ms, 2),
            candidates=len(chunks),
            returned=len(result),
        )

        return result
