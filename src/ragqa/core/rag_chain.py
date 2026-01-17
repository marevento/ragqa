"""RAG orchestration - combines retrieval with LLM generation."""

import re
from collections.abc import Generator
from dataclasses import dataclass

from ragqa.core.models import Chunk
from ragqa.llm.client import generate
from ragqa.llm.prompts import (
    ALL_DOCS_PROMPT,
    SINGLE_DOC_PROMPT,
    SPECIFIC_QUERY_PROMPT,
    build_context,
)
from ragqa.retrieval.bm25_index import BM25Index
from ragqa.retrieval.query_classifier import QueryType, classify_query
from ragqa.retrieval.retriever import Retriever
from ragqa.retrieval.vectorstore import VectorStore

MAX_CONTEXT_CHARS = 20000  # ~5000 tokens


@dataclass
class RAGResponse:
    """Response from RAG chain with sources."""

    answer: str
    sources: list[dict[str, str | int]]
    confidence: int
    query_type: QueryType


class RAGChain:
    """Orchestrates RAG pipeline: classify → retrieve → generate."""

    def __init__(
        self,
        vectorstore: VectorStore | None = None,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self.vectorstore = vectorstore or VectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.retriever = Retriever(self.vectorstore, self.bm25_index)

    def ask(
        self, question: str, stream: bool = False
    ) -> RAGResponse | Generator[str, None, RAGResponse]:
        """Answer a question using RAG."""
        query_type = classify_query(question)

        if query_type == "all_docs":
            return self._handle_all_docs(stream)
        elif query_type == "single_doc":
            return self._handle_single_doc(question, stream)
        else:
            return self._handle_specific(question, stream)

    def _handle_specific(
        self, question: str, stream: bool
    ) -> RAGResponse | Generator[str, None, RAGResponse]:
        """Handle specific factual/how-to queries."""
        chunks = self.retriever.retrieve(question)
        confidence = self.retriever.calculate_confidence(chunks)

        if not chunks:
            return RAGResponse(
                answer="No relevant content found. Try rephrasing your question.",
                sources=[],
                confidence=0,
                query_type="specific",
            )

        # Truncate to context window and extract sources from used chunks only
        used_chunks = self._truncate_context(chunks)
        context_items = [(c.filename, c.title, c.page, c.text) for c in used_chunks]
        context = build_context(context_items)

        prompt = SPECIFIC_QUERY_PROMPT.format(context=context, question=question)

        # Sources that were actually sent to the LLM
        context_sources = self._extract_sources(used_chunks)

        if stream:
            return self._stream_with_response(
                prompt, context_sources, confidence, "specific"
            )
        else:
            answer = str(generate(prompt, stream=False))
            answer = self._clean_response(answer)
            return RAGResponse(
                answer=answer,
                sources=context_sources,
                confidence=confidence,
                query_type="specific",
            )

    def _handle_all_docs(
        self, stream: bool
    ) -> RAGResponse | Generator[str, None, RAGResponse]:
        """Handle all-docs summarization queries."""
        documents = self.vectorstore.get_all_documents()

        if not documents:
            return RAGResponse(
                answer="No documents indexed. Run 'ragqa index' first.",
                sources=[],
                confidence=0,
                query_type="all_docs",
            )

        # Build titles list
        titles = []
        for doc in documents:
            meta = doc.get("metadata", {})
            filename = meta.get("filename", doc.get("id", "unknown"))
            title = meta.get("title", "Untitled")
            titles.append(f"- {filename}: {title}")

        titles_text = "\n".join(titles)
        prompt = ALL_DOCS_PROMPT.format(titles=titles_text)

        sources = [
            {
                "filename": doc.get("metadata", {}).get("filename", ""),
                "title": doc.get("metadata", {}).get("title", ""),
                "authors": doc.get("metadata", {}).get("authors", "Unknown"),
            }
            for doc in documents
        ]

        if stream:
            return self._stream_with_response(prompt, sources, 100, "all_docs")
        else:
            answer = str(generate(prompt, stream=False))
            answer = self._clean_response(answer)
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=100,
                query_type="all_docs",
            )

    def _handle_single_doc(
        self, question: str, stream: bool
    ) -> RAGResponse | Generator[str, None, RAGResponse]:
        """Handle single document summarization."""
        # Extract document ID from question
        match = re.search(r"(2510\.\d+)", question)
        if not match:
            return self._handle_specific(question, stream)

        doc_id = match.group(1)

        # Find document by ID
        documents = self.vectorstore.get_all_documents()
        target_doc = None
        for doc in documents:
            if doc_id in doc.get("id", "") or doc_id in doc.get("metadata", {}).get(
                "filename", ""
            ):
                target_doc = doc
                break

        if not target_doc:
            return RAGResponse(
                answer=f"Document {doc_id} not found in index.",
                sources=[],
                confidence=0,
                query_type="single_doc",
            )

        meta = target_doc.get("metadata", {})
        filename = meta.get("filename", "")
        context = target_doc.get("text", "")

        prompt = SINGLE_DOC_PROMPT.format(
            context=context[:MAX_CONTEXT_CHARS], filename=filename
        )

        sources = [
            {
                "filename": filename,
                "title": meta.get("title", ""),
                "authors": meta.get("authors", "Unknown"),
            }
        ]

        if stream:
            return self._stream_with_response(prompt, sources, 95, "single_doc")
        else:
            answer = str(generate(prompt, stream=False))
            answer = self._clean_response(answer)
            return RAGResponse(
                answer=answer,
                sources=sources,
                confidence=95,
                query_type="single_doc",
            )

    def _stream_with_response(
        self,
        prompt: str,
        sources: list[dict[str, str | int]],
        confidence: int,
        query_type: QueryType,
    ) -> Generator[str, None, RAGResponse]:
        """Stream response tokens and return final RAGResponse."""
        full_response = ""
        gen = generate(prompt, stream=True)
        if isinstance(gen, str):
            full_response = gen
            yield gen
        else:
            for token in gen:
                full_response += token
                yield token

        return RAGResponse(
            answer=self._clean_response(full_response),
            sources=sources,
            confidence=confidence,
            query_type=query_type,
        )

    def _truncate_context(self, chunks: list[Chunk]) -> list[Chunk]:
        """Truncate chunks to fit context window."""
        result: list[Chunk] = []
        total = 0
        for chunk in chunks:
            if total + len(chunk.text) > MAX_CONTEXT_CHARS:
                break
            result.append(chunk)
            total += len(chunk.text)
        return result

    def _extract_sources(self, chunks: list[Chunk]) -> list[dict[str, str | int]]:
        """Extract unique sources from chunks."""
        seen: set[str] = set()
        sources: list[dict[str, str | int]] = []

        for chunk in chunks:
            filename = chunk.filename
            if filename and filename not in seen:
                seen.add(filename)
                sources.append(
                    {
                        "filename": filename,
                        "title": chunk.title,
                        "authors": chunk.authors,
                        "page": chunk.page,
                    }
                )

        return sources

    def _clean_response(self, response: str) -> str:
        """Remove verbose preambles from LLM response."""
        preambles = [
            r"^(Sure!|Of course!|Here's|Based on|I'd be happy to)[^.]*\.\s*",
            r"^(The context|The documents?|According to the)[^:]*:\s*",
        ]
        for pattern in preambles:
            response = re.sub(pattern, "", response, flags=re.IGNORECASE)
        return response.strip()
