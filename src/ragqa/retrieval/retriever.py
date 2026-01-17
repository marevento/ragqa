"""Hybrid retrieval orchestration with RRF fusion."""

from ragqa.config import get_settings
from ragqa.core.models import Chunk
from ragqa.retrieval.bm25_index import BM25Index
from ragqa.retrieval.embeddings import get_embedding
from ragqa.retrieval.vectorstore import VectorStore

# Title boost factor for RRF
TITLE_BOOST_WEIGHT = 2.0

# Score filtering thresholds
MIN_SCORE_RATIO = 0.5  # Drop results with score < 50% of top score
SCORE_GAP_THRESHOLD = 0.3  # Drop if score drops by >30% from previous


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
        """Retrieve relevant chunks using hybrid search with title boosting."""
        settings = get_settings()
        k = top_k or settings.retrieval_top_k

        # Get results from both sources
        fetch_k = k * 3

        semantic_results = self.vectorstore.search_chunks(query, fetch_k)
        bm25_results = self.bm25_index.search(query, fetch_k)

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

        # Filter out low-scoring results
        filtered = self._filter_by_score(merged[:k])
        return filtered

    def _compute_title_scores(self, query: str) -> dict[str, float]:
        """Compute similarity between query and document titles."""
        import numpy as np

        query_emb = get_embedding(query)
        docs = self.vectorstore.get_all_documents()
        title_scores: dict[str, float] = {}

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
        """Filter out chunks with scores significantly lower than the top result."""
        if not chunks:
            return []

        top_score = chunks[0].score
        min_score = top_score * MIN_SCORE_RATIO

        result: list[Chunk] = []
        prev_score = top_score

        for chunk in chunks:
            # Drop if below minimum threshold
            if chunk.score < min_score:
                break

            # Drop if score gap is too large
            if prev_score > 0:
                gap = (prev_score - chunk.score) / prev_score
                if gap > SCORE_GAP_THRESHOLD and len(result) > 0:
                    break

            result.append(chunk)
            prev_score = chunk.score

        return result

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
