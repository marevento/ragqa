"""BM25 keyword index for hybrid retrieval."""

import json
import re
from pathlib import Path
from typing import Any

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

        self.data_path = self.persist_dir / "bm25_data.json"

        self.bm25: BM25Okapi | None = None
        self.chunks: list[Chunk] = []
        self.corpus: list[list[str]] = []

        self._load()

    def _load(self) -> None:
        """Load index from disk and rebuild BM25 from corpus."""
        if self.data_path.exists():
            try:
                with open(self.data_path) as f:
                    data: dict[str, Any] = json.load(f)
                self.corpus = data.get("corpus", [])
                self.chunks = [
                    Chunk(**chunk_data) for chunk_data in data.get("chunks", [])
                ]
                if self.corpus:
                    self.bm25 = BM25Okapi(self.corpus)
            except Exception:
                self.bm25 = None
                self.chunks = []
                self.corpus = []

        # Migrate legacy pickle files if JSON doesn't exist
        elif self._migrate_legacy():
            pass

    def _migrate_legacy(self) -> bool:
        """Migrate legacy pickle files to JSON format."""
        legacy_index = self.persist_dir / "bm25_index.pkl"
        legacy_chunks = self.persist_dir / "bm25_chunks.pkl"

        if not (legacy_index.exists() and legacy_chunks.exists()):
            return False

        try:
            import pickle  # noqa: S403 — one-time migration only

            with open(legacy_chunks, "rb") as f:
                data = pickle.load(f)  # noqa: S301
            self.chunks = data.get("chunks", [])
            self.corpus = data.get("corpus", [])
            if self.corpus:
                self.bm25 = BM25Okapi(self.corpus)
                self._save()
            # Remove legacy files after successful migration
            legacy_index.unlink(missing_ok=True)
            legacy_chunks.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def _save(self) -> None:
        """Save chunks and corpus to disk as JSON."""
        if self.corpus:
            data = {
                "chunks": [chunk.model_dump() for chunk in self.chunks],
                "corpus": self.corpus,
            }
            with open(self.data_path, "w") as f:
                json.dump(data, f)

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
        """Add a single document to the index."""
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
        top_score = float(max(scores)) if len(scores) > 0 else 0.0
        max_score = top_score if top_score > 0 else 1.0

        for idx, score in scored_indices:
            if score > 0:
                chunk = self.chunks[idx]
                normalized_score = float(score) / max_score
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
        if self.data_path.exists():
            self.data_path.unlink()
        # Clean up legacy files if present
        for legacy in ("bm25_index.pkl", "bm25_chunks.pkl"):
            path = self.persist_dir / legacy
            if path.exists():
                path.unlink()

    def is_indexed(self) -> bool:
        """Check if index has been built."""
        return self.bm25 is not None and len(self.chunks) > 0

    def chunk_count(self) -> int:
        """Get number of indexed chunks."""
        return len(self.chunks)
