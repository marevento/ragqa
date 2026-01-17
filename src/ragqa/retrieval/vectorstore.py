"""ChromaDB vector store wrapper."""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from ragqa.config import get_settings
from ragqa.core.models import Chunk, Document
from ragqa.exceptions import IndexError as RAGIndexError
from ragqa.retrieval.embeddings import get_embedding, get_embeddings_batch


class VectorStore:
    """ChromaDB-backed vector store for document chunks."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        settings = get_settings()
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Collection for content chunks
        self.chunks_collection = self.client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
        )

        # Collection for document-level entries (title + abstract)
        self.documents_collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(self, document: Document) -> None:
        """Add a document and its chunks to the vector store."""
        # Add document-level entry
        doc_text = f"{document.title}\n{' '.join(document.title_variants)}\n\n{document.abstract}"
        doc_embedding = get_embedding(doc_text)

        self.documents_collection.upsert(
            ids=[document.id],
            embeddings=[doc_embedding],  # type: ignore[arg-type]
            documents=[doc_text],
            metadatas=[
                {
                    "filename": document.filename,
                    "title": document.title,
                    "authors": document.authors,
                    "page_count": document.page_count,
                }
            ],
        )

        # Add chunks in batches
        if document.chunks:
            chunk_ids = [c.id for c in document.chunks]
            chunk_texts = [c.text for c in document.chunks]
            chunk_metadatas: list[dict[str, Any]] = []
            for c in document.chunks:
                page_val = c.metadata.get("page", 0)
                page_num = int(page_val) if isinstance(page_val, int | str) else 0
                meta: dict[str, Any] = {
                    "filename": c.metadata.get("filename", ""),
                    "title": c.metadata.get("title", ""),
                    "authors": c.metadata.get("authors", ""),
                    "page": page_num,
                }
                if "chunk_type" in c.metadata:
                    meta["chunk_type"] = str(c.metadata["chunk_type"])
                if "chunk_index" in c.metadata:
                    idx_val = c.metadata["chunk_index"]
                    meta["chunk_index"] = int(idx_val) if isinstance(idx_val, int | str) else 0
                chunk_metadatas.append(meta)

            # Batch embeddings for efficiency
            chunk_embeddings = get_embeddings_batch(chunk_texts)

            self.chunks_collection.upsert(
                ids=chunk_ids,
                embeddings=chunk_embeddings,  # type: ignore[arg-type]
                documents=chunk_texts,
                metadatas=chunk_metadatas,  # type: ignore[arg-type]
            )

    def search_chunks(self, query: str, top_k: int | None = None) -> list[Chunk]:
        """Search for similar chunks."""
        settings = get_settings()
        k = top_k or settings.retrieval_top_k

        if self.chunks_collection.count() == 0:
            raise RAGIndexError(
                message="No documents indexed",
                details="Run 'ragqa index' first to build the index.",
            )

        query_embedding = get_embedding(query)

        results = self.chunks_collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=k,
            include=["documents", "metadatas", "distances"],  # type: ignore[list-item]
        )

        chunks: list[Chunk] = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                # Convert cosine distance to similarity score
                score = 1.0 - distance

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                text = results["documents"][0][i] if results["documents"] else ""

                chunks.append(
                    Chunk(
                        id=chunk_id,
                        text=text,
                        metadata=metadata,
                        score=score,
                    )
                )

        return chunks

    def search_documents(
        self, query: str, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Search for similar documents (title + abstract level)."""
        settings = get_settings()
        k = top_k or settings.retrieval_top_k

        if self.documents_collection.count() == 0:
            return []

        query_embedding = get_embedding(query)

        results = self.documents_collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=k,
            include=["documents", "metadatas", "distances"],  # type: ignore[list-item]
        )

        documents: list[dict[str, Any]] = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0.0
                score = 1.0 - distance

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                text = results["documents"][0][i] if results["documents"] else ""

                documents.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "metadata": metadata,
                        "score": score,
                    }
                )

        return documents

    def get_all_documents(self) -> list[dict[str, Any]]:
        """Get all indexed documents."""
        if self.documents_collection.count() == 0:
            return []

        results = self.documents_collection.get(include=["documents", "metadatas"])  # type: ignore[list-item]

        documents: list[dict[str, Any]] = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                text = results["documents"][i] if results["documents"] else ""

                documents.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "metadata": metadata,
                    }
                )

        return documents

    def document_count(self) -> int:
        """Get number of indexed documents."""
        return self.documents_collection.count()

    def chunk_count(self) -> int:
        """Get number of indexed chunks."""
        return self.chunks_collection.count()

    def clear(self) -> None:
        """Clear all collections."""
        self.client.delete_collection("chunks")
        self.client.delete_collection("documents")
        self.chunks_collection = self.client.get_or_create_collection(
            name="chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self.documents_collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"},
        )

    def is_indexed(self) -> bool:
        """Check if any documents have been indexed."""
        return self.documents_collection.count() > 0
