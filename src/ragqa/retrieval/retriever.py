"""Hybrid retrieval orchestration with RRF fusion."""

from ragqa.config import get_settings
from ragqa.core.models import Chunk
from ragqa.retrieval.bm25_index import BM25Index
from ragqa.retrieval.vectorstore import VectorStore


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
        vectorstore: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.vectorstore = vectorstore or VectorStore()
        self.bm25_index = bm25_index or BM25Index()

    def retrieve(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Retrieve relevant chunks using hybrid search."""
        settings = get_settings()
        k = top_k or settings.retrieval_top_k

        # Get results from both sources
        # Retrieve more than top_k from each to allow RRF to work well
        fetch_k = k * 3

        semantic_results = self.vectorstore.search_chunks(query, fetch_k)
        bm25_results = self.bm25_index.search(query, fetch_k)

        # Merge using RRF
        if bm25_results and semantic_results:
            merged = reciprocal_rank_fusion([semantic_results, bm25_results])
        elif semantic_results:
            merged = semantic_results
        elif bm25_results:
            merged = bm25_results
        else:
            merged = []

        return merged[:k]

    def retrieve_semantic_only(
        self, query: str, top_k: int | None = None
    ) -> list[Chunk]:
        """Retrieve using only semantic search."""
        settings = get_settings()
        k = top_k or settings.retrieval_top_k
        return self.vectorstore.search_chunks(query, k)

    def retrieve_bm25_only(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Retrieve using only BM25 keyword search."""
        settings = get_settings()
        k = top_k or settings.retrieval_top_k
        return self.bm25_index.search(query, k)

    def calculate_confidence(self, chunks: list[Chunk]) -> int:
        """Calculate answer confidence from retrieval scores."""
        if not chunks:
            return 0
        top_scores = sorted([c.score for c in chunks], reverse=True)[:3]
        avg_score = sum(top_scores) / len(top_scores)
        # Scale RRF scores to percentage (RRF scores are typically small)
        # Max RRF score for rank 1 with k=60 is 1/61 ≈ 0.016
        # With 2 sources, max is about 0.032
        confidence = min(100, int(avg_score * 3000))
        return confidence
