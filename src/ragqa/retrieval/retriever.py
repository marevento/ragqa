"""Hybrid retrieval orchestration with RRF fusion."""

import asyncio
import time

from ragqa import get_logger
from ragqa.config import get_settings
from ragqa.core.models import Chunk
from ragqa.protocols import KeywordIndex, RerankerProtocol, VectorStoreProtocol
from ragqa.retrieval.bm25_index import BM25Index
from ragqa.retrieval.embeddings import get_embedding, get_embedding_async
from ragqa.retrieval.vectorstore import VectorStore

logger = get_logger(__name__)

# Title boost factor for RRF
TITLE_BOOST_WEIGHT = 2.0

# Document-level filtering thresholds
DOC_SCORE_RATIO = 0.7  # Drop documents with score < 70% of top document

# Max candidates to send to cross-encoder reranker (limits latency)
RERANK_CANDIDATE_LIMIT = 3  # multiplier of top_k


def reciprocal_rank_fusion(results_list: list[list[Chunk]], k: int = 60) -> list[Chunk]:
    """Merge multiple result lists using Reciprocal Rank Fusion."""
    # Map chunk ID to (chunk, cumulative RRF score)
    scores: dict[str, tuple[Chunk, float]] = {}

    for results in results_list:
        for rank, chunk in enumerate(results):
            rrf_score = 1.0 / (k + rank + 1)
            if chunk.id in scores:
                existing_chunk, existing_score = scores[chunk.id]
                # Keep chunk with higher original score, sum RRF scores
                if chunk.score > existing_chunk.score:
                    scores[chunk.id] = (chunk, existing_score + rrf_score)
                else:
                    scores[chunk.id] = (existing_chunk, existing_score + rrf_score)
            else:
                scores[chunk.id] = (chunk, rrf_score)

    # Sort by RRF score and return chunks with updated scores
    sorted_items = sorted(scores.items(), key=lambda x: x[1][1], reverse=True)

    result: list[Chunk] = []
    for _chunk_id, (chunk, rrf_score) in sorted_items:
        result.append(
            Chunk(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=rrf_score,  # Use RRF score as final score
            )
        )

    return result


class Retriever:
    """Hybrid retriever combining BM25 and semantic search."""

    def __init__(
        self,
        vectorstore: VectorStoreProtocol | None = None,
        bm25_index: KeywordIndex | None = None,
        reranker: RerankerProtocol | None = None,
    ) -> None:
        self.vectorstore: VectorStoreProtocol = vectorstore or VectorStore()
        self.bm25_index: KeywordIndex = bm25_index or BM25Index()
        self.reranker = reranker

    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Retrieve relevant chunks using hybrid search with title boosting."""
        start_time = time.time()
        settings = get_settings()
        k = top_k or settings.retrieval_top_k

        logger.debug("retrieval_started", query=query[:50], top_k=k)

        # Get results from both sources
        fetch_k = k * 3

        semantic_results = self.vectorstore.search_chunks(query, fetch_k)
        bm25_results = self.bm25_index.search(query, fetch_k)

        logger.debug(
            "retrieval_sources_fetched",
            semantic_count=len(semantic_results),
            bm25_count=len(bm25_results),
        )

        # Compute title relevance scores for each document
        title_scores = self._compute_title_scores(query)

        # Boost chunks based on title relevance
        boosted_semantic = self._apply_title_boost(semantic_results, title_scores)
        boosted_bm25 = self._apply_title_boost(bm25_results, title_scores)

        # Merge using RRF
        if boosted_bm25 and boosted_semantic:
            merged = reciprocal_rank_fusion([boosted_semantic, boosted_bm25])
        elif boosted_semantic:
            merged = boosted_semantic
        elif boosted_bm25:
            merged = boosted_bm25
        else:
            merged = []

        # Re-rank with cross-encoder if available
        reranked = False
        if self.reranker is not None and merged:
            rerank_limit = k * RERANK_CANDIDATE_LIMIT
            merged = self.reranker.rerank(query, merged[:rerank_limit], k)
            reranked = True

        # Skip _filter_by_score after reranking — sigmoid scores are in a
        # different domain than RRF scores, and the reranker already
        # selected the top-k most relevant chunks.
        filtered = merged[:k] if reranked else self._filter_by_score(merged[:k])

        duration_ms = (time.time() - start_time) * 1000
        logger.info(
            "retrieval_completed",
            duration_ms=round(duration_ms, 2),
            chunks_returned=len(filtered),
            query_preview=query[:30],
        )

        return filtered

    def _compute_title_scores(self, query: str) -> dict[str, float]:
        """Compute similarity between query and document titles.

        Uses cached title embeddings from VectorStore for efficiency.
        Falls back to computing embeddings if cache is unavailable.
        """
        import numpy as np

        query_emb = get_embedding(query)
        title_scores: dict[str, float] = {}

        # Try to use cached title embeddings first
        cached_embeddings = self.vectorstore.get_title_embeddings()

        if cached_embeddings:
            # Use cached embeddings for efficiency
            for filename, title_emb in cached_embeddings.items():
                # Cosine similarity
                similarity = float(
                    np.dot(query_emb, title_emb)
                    / (np.linalg.norm(query_emb) * np.linalg.norm(title_emb))
                )
                title_scores[filename] = similarity
        else:
            # Fallback: compute embeddings on the fly (for backward compatibility)
            docs = self.vectorstore.get_all_documents()
            for doc in docs:
                meta = doc.get("metadata", {})
                title = meta.get("title", "")
                filename = meta.get("filename", "")

                if title and filename:
                    title_emb = get_embedding(title)
                    # Cosine similarity
                    similarity = float(
                        np.dot(query_emb, title_emb)
                        / (np.linalg.norm(query_emb) * np.linalg.norm(title_emb))
                    )
                    title_scores[filename] = similarity

        return title_scores

    def _apply_title_boost(
        self, chunks: list[Chunk], title_scores: dict[str, float]
    ) -> list[Chunk]:
        """Boost chunk scores based on title relevance."""
        boosted: list[Chunk] = []
        for chunk in chunks:
            title_sim = title_scores.get(chunk.filename, 0.0)
            # Apply boost if title is relevant (similarity > 0.5)
            if title_sim > 0.5:
                boost = 1.0 + (title_sim - 0.5) * TITLE_BOOST_WEIGHT
                new_score = chunk.score * boost
            else:
                new_score = chunk.score

            boosted.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=new_score,
                )
            )

        # Re-sort by boosted score
        boosted.sort(key=lambda c: c.score, reverse=True)
        return boosted

    def _filter_by_score(self, chunks: list[Chunk]) -> list[Chunk]:
        """Filter chunks to keep only those from top-scoring documents."""
        if not chunks:
            return []

        # Group chunks by document and sum scores
        doc_scores: dict[str, float] = {}
        for chunk in chunks:
            doc_scores[chunk.filename] = doc_scores.get(chunk.filename, 0) + chunk.score

        # Find top document score
        top_doc_score = max(doc_scores.values())
        min_doc_score = top_doc_score * DOC_SCORE_RATIO

        # Keep only documents above threshold
        valid_docs = {doc for doc, score in doc_scores.items() if score >= min_doc_score}

        # Filter chunks to only those from valid documents
        return [c for c in chunks if c.filename in valid_docs]

    def calculate_confidence(self, chunks: list[Chunk]) -> int:
        """Calculate answer confidence from retrieval scores.

        Handles two score domains:
        - RRF scores (~0.01-0.03): scaled by 3000x to reach 0-100
        - Reranker sigmoid scores (0.0-1.0): scaled by 100x to reach 0-100

        The heuristic threshold is 0.1: scores above that are assumed to
        be sigmoid-normalized (reranker output), scores below are RRF.
        """
        if not chunks:
            return 0
        top_scores = sorted([c.score for c in chunks], reverse=True)[:3]
        avg_score = sum(top_scores) / len(top_scores)

        if avg_score > 0.1:
            # Sigmoid-normalized scores from cross-encoder (0.0-1.0)
            confidence = min(100, int(avg_score * 100))
        else:
            # RRF scores — max is ~0.032 for rank-1 with 2 sources
            confidence = min(100, int(avg_score * 3000))
        return confidence

    async def retrieve_async(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Retrieve relevant chunks using hybrid search with title boosting (async).

        Args:
            query: The search query.
            top_k: Maximum number of results to return.

        Returns:
            List of relevant chunks with scores.
        """
        settings = get_settings()
        k = top_k or settings.retrieval_top_k

        # Get results from both sources
        fetch_k = k * 3

        # Async semantic search, sync BM25 search
        # Note: Type ignore needed because protocol doesn't include async methods
        semantic_results = await self.vectorstore.search_chunks_async(query, fetch_k)  # type: ignore[attr-defined]
        bm25_results = self.bm25_index.search(query, fetch_k)

        # Compute title relevance scores for each document (async)
        title_scores = await self._compute_title_scores_async(query)

        # Boost chunks based on title relevance
        boosted_semantic = self._apply_title_boost(semantic_results, title_scores)
        boosted_bm25 = self._apply_title_boost(bm25_results, title_scores)

        # Merge using RRF
        if boosted_bm25 and boosted_semantic:
            merged = reciprocal_rank_fusion([boosted_semantic, boosted_bm25])
        elif boosted_semantic:
            merged = boosted_semantic
        elif boosted_bm25:
            merged = boosted_bm25
        else:
            merged = []

        # Re-rank with cross-encoder if available
        # Run in thread to avoid blocking the event loop (CPU-bound)
        reranked = False
        if self.reranker is not None and merged:
            rerank_limit = k * RERANK_CANDIDATE_LIMIT
            merged = await asyncio.to_thread(
                self.reranker.rerank, query, merged[:rerank_limit], k
            )
            reranked = True

        # Skip _filter_by_score after reranking (see sync path comment)
        filtered = merged[:k] if reranked else self._filter_by_score(merged[:k])
        return filtered

    async def _compute_title_scores_async(self, query: str) -> dict[str, float]:
        """Compute similarity between query and document titles (async).

        Uses cached title embeddings from VectorStore for efficiency.
        Falls back to computing embeddings if cache is unavailable.
        """
        import numpy as np

        query_emb = await get_embedding_async(query)
        title_scores: dict[str, float] = {}

        # Try to use cached title embeddings first
        cached_embeddings = self.vectorstore.get_title_embeddings()

        if cached_embeddings:
            # Use cached embeddings for efficiency
            for filename, title_emb in cached_embeddings.items():
                # Cosine similarity
                similarity = float(
                    np.dot(query_emb, title_emb)
                    / (np.linalg.norm(query_emb) * np.linalg.norm(title_emb))
                )
                title_scores[filename] = similarity
        else:
            # Fallback: compute embeddings on the fly (for backward compatibility)
            docs = self.vectorstore.get_all_documents()
            for doc in docs:
                meta = doc.get("metadata", {})
                title = meta.get("title", "")
                filename = meta.get("filename", "")

                if title and filename:
                    title_emb = await get_embedding_async(title)
                    # Cosine similarity
                    similarity = float(
                        np.dot(query_emb, title_emb)
                        / (np.linalg.norm(query_emb) * np.linalg.norm(title_emb))
                    )
                    title_scores[filename] = similarity

        return title_scores

