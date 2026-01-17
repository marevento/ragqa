"""Pydantic data models for RAG Q&A."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A text chunk from a document."""

    id: str
    text: str
    metadata: dict[str, str | int | list[str]] = Field(default_factory=dict)
    score: float = 0.0

    @property
    def filename(self) -> str:
        return str(self.metadata.get("filename", ""))

    @property
    def page(self) -> int:
        page_val = self.metadata.get("page", 0)
        if isinstance(page_val, int):
            return page_val
        if isinstance(page_val, str):
            return int(page_val) if page_val else 0
        return 0

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))

    @property
    def authors(self) -> str:
        return str(self.metadata.get("authors", "Unknown"))


class Document(BaseModel):
    """A parsed document with metadata."""

    id: str
    filename: str
    title: str
    title_variants: list[str] = Field(default_factory=list)
    authors: str = "Unknown"
    abstract: str = ""
    page_count: int = 0
    chunks: list[Chunk] = Field(default_factory=list)


class SearchResult(BaseModel):
    """A search result with source information."""

    chunks: list[Chunk]
    query_type: str = "specific"
