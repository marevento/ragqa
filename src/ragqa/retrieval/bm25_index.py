"""BM25 keyword index for hybrid retrieval."""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from ragqa.config import get_settings
from ragqa.core.models import Chunk, Document


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    text = text.lower()
    # Split on non-alphanumeric characters
    tokens = re.findall(r"\b\w+\b", text)
    return tokens


class BM25Index:
    """BM25 keyword search index."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.persist_dir / "bm25_index.pkl"
        self.chunks_path = self.persist_dir / "bm25_chunks.pkl"

        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []
        self.corpus: list[list[str]] = []

        self._load()

    def _load(self) -> None:
        """Load index from disk if exists."""
        if self.index_path.exists() and self.chunks_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                with open(self.chunks_path, "rb") as f:
                    data = pickle.load(f)
                    self.chunks = data.get("chunks", [])
                    self.corpus = data.get("corpus", [])
            except Exception:
                self.bm25 = None
                self.chunks = []
                self.corpus = []

    def _save(self) -> None:
        """Save index to disk."""
        if self.bm25 is not None:
            with open(self.index_path, "wb") as f:
                pickle.dump(self.bm25, f)
            with open(self.chunks_path, "wb") as f:
                pickle.dump({"chunks": self.chunks, "corpus": self.corpus}, f)

    def build_from_documents(self, documents: list[Document]) -> None:
        """Build BM25 index from documents."""
        self.chunks = []
        self.corpus = []

        for doc in documents:
            for chunk in doc.chunks:
                # Tokenize with both original case and lowercase for better recall
                tokens = tokenize(chunk.text)
                # Also add title tokens for better matching
                title_tokens = tokenize(chunk.title)
                all_tokens = tokens + title_tokens

                self.chunks.append(chunk)
                self.corpus.append(all_tokens)

        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            self._save()

    def add_document(self, document: Document) -> None:
        """Add a single document to the index.

        Note: For better performance when adding multiple documents,
        use add_documents_batch() instead.
        """
        for chunk in document.chunks:
            tokens = tokenize(chunk.text)
            title_tokens = tokenize(chunk.title)
            all_tokens = tokens + title_tokens

            self.chunks.append(chunk)
            self.corpus.append(all_tokens)

        # Rebuild index with new documents
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            self._save()

    def add_documents_batch(self, documents: list[Document]) -> None:
        """Add multiple documents and rebuild index once.

        This is more efficient than calling add_document() multiple times
        because it only rebuilds the BM25 index once after all documents
        are added.

        Args:
            documents: List of documents to add.
        """
        for doc in documents:
            for chunk in doc.chunks:
                tokens = tokenize(chunk.text)
                title_tokens = tokenize(chunk.title)
                all_tokens = tokens + title_tokens

                self.chunks.append(chunk)
                self.corpus.append(all_tokens)

        # Single rebuild at end
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
            self._save()

    def search(self, query: str, top_k: int = 10) -> list[Chunk]:
        """Search for chunks matching the query."""
        if self.bm25 is None or not self.chunks:
            return []

        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        results: list[Chunk] = []
        for idx, score in scored_indices:
            if score > 0:
                chunk = self.chunks[idx]
                # Create new chunk with BM25 score normalized to 0-1 range
                max_score = max(scores) if max(scores) > 0 else 1
                normalized_score = score / max_score
                results.append(
                    Chunk(
                        id=chunk.id,
                        text=chunk.text,
                        metadata=chunk.metadata,
                        score=normalized_score,
                    )
                )

        return results

    def clear(self) -> None:
        """Clear the index."""
        self.bm25 = None
        self.chunks = []
        self.corpus = []
        if self.index_path.exists():
            self.index_path.unlink()
        if self.chunks_path.exists():
            self.chunks_path.unlink()

    def is_indexed(self) -> bool:
        """Check if index has been built."""
        return self.bm25 is not None and len(self.chunks) > 0

    def chunk_count(self) -> int:
        """Get number of indexed chunks."""
        return len(self.chunks)
