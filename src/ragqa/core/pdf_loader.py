"""PDF parsing and chunking module."""

import re
from pathlib import Path

import fitz

from ragqa.config import get_settings
from ragqa.core.models import Chunk, Document
from ragqa.exceptions import PDFError


def extract_text_reading_order(page: fitz.Page) -> str:
    """Extract text respecting multi-column layout."""
    blocks = page.get_text("dict")["blocks"]
    text_blocks: list[tuple[float, float, str]] = []

    for block in blocks:
        if block.get("type") == 0:  # Text block
            bbox = block["bbox"]
            text_parts: list[str] = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text_parts.append(span.get("text", ""))
            text = " ".join(text_parts)
            if text.strip():
                text_blocks.append((bbox[1], bbox[0], text))

    # Sort by y first (top to bottom), then x (left to right within row)
    text_blocks.sort(key=lambda b: (b[0] // 50, b[1]))
    return "\n".join(b[2] for b in text_blocks)


def extract_title(
    doc: fitz.Document, first_page_text: str
) -> dict[str, str | list[str]]:
    """Extract title from PDF metadata and document text."""
    metadata_title = (doc.metadata.get("title") or "").strip()

    # Extract first significant line from document text
    lines = first_page_text.strip().split("\n")
    doc_title = ""
    for line in lines[:10]:
        line = line.strip()
        # Skip short lines, numbers, dates
        if len(line) > 20 and not line.isdigit() and not re.match(r"^\d{4}", line):
            doc_title = line
            break

    title = metadata_title or doc_title or "Untitled"
    variants = list({metadata_title, doc_title} - {""})

    return {"title": title, "title_variants": variants if variants else [title]}


def extract_authors(doc: fitz.Document, first_page_text: str) -> str:
    """Extract authors from PDF metadata or text."""
    # Try PDF metadata first
    if doc.metadata.get("author"):
        return format_authors(doc.metadata["author"])

    # Fallback: look for author patterns after title
    author_match = re.search(
        r"(?:^|\n)([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+"
        r"(?:,?\s*(?:and\s+|&\s*)?[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)*)",
        first_page_text[:2000],
    )
    if author_match:
        return format_authors(author_match.group(1))

    return "Unknown"


def format_authors(authors: str) -> str:
    """Format as 'LastName et al.' if multiple authors."""
    names = re.split(r",|;|\s+and\s+|\s*&\s*", authors)
    names = [n.strip() for n in names if n.strip()]
    if len(names) > 2:
        first_last = names[0].split()[-1] if names[0].split() else names[0]
        return f"{first_last} et al."
    return authors.strip()[:50]


def extract_abstract(doc: fitz.Document) -> str:
    """Extract abstract from first 2 pages, handling images and multi-column."""
    for page_num in range(min(2, len(doc))):
        page = doc[page_num]
        blocks = page.get_text("blocks")
        text_blocks = [b[4] for b in blocks if b[6] == 0]
        text = "\n".join(text_blocks)

        abstract_match = re.search(
            r"\bAbstract\b[:\.\s]*(.{100,2000}?)"
            r"(?=\n\s*(?:Introduction|Keywords|1\s*[\.\)]|I\s*\.)|\Z)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            abstract = re.sub(r"\s+", " ", abstract)
            return abstract[:2000]

    return ""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < text_len:
            # Look for sentence end near chunk boundary
            for sep in [". ", ".\n", "? ", "! "]:
                last_sep = text.rfind(sep, start + chunk_size // 2, end)
                if last_sep != -1:
                    end = last_sep + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= text_len:
            break

    return chunks


def load_pdf(pdf_path: Path) -> Document:
    """Load and parse a PDF file into a Document."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PDFError(
            message=f"Cannot open PDF: {pdf_path.name}",
            details=str(e),
        ) from e

    settings = get_settings()
    filename = pdf_path.name
    doc_id = pdf_path.stem

    # Extract text from all pages
    pages_text: list[str] = []
    for page in doc:
        pages_text.append(extract_text_reading_order(page))

    full_text = "\n\n".join(pages_text)
    first_page_text = pages_text[0] if pages_text else ""

    # Extract metadata
    title_info = extract_title(doc, first_page_text)
    title = str(title_info["title"])
    title_variants = list(title_info.get("title_variants", [title]))
    authors = extract_authors(doc, first_page_text)
    abstract = extract_abstract(doc)

    # Create chunks
    text_chunks = chunk_text(full_text, settings.chunk_size, settings.chunk_overlap)

    chunks: list[Chunk] = []

    # Add title chunk for better title matching
    title_text = f"{title}\n" + "\n".join(title_variants)
    chunks.append(
        Chunk(
            id=f"{doc_id}_title",
            text=title_text,
            metadata={
                "filename": filename,
                "title": title,
                "authors": authors,
                "page": 1,
                "chunk_type": "title",
            },
        )
    )

    # Add abstract chunk if available
    if abstract:
        chunks.append(
            Chunk(
                id=f"{doc_id}_abstract",
                text=f"{title}\n\n{abstract}",
                metadata={
                    "filename": filename,
                    "title": title,
                    "authors": authors,
                    "page": 1,
                    "chunk_type": "abstract",
                },
            )
        )

    # Add content chunks with page estimation
    chars_per_page = len(full_text) // max(len(doc), 1)
    for i, chunk_text_content in enumerate(text_chunks):
        # Estimate page number based on position
        chunk_start = sum(len(c) for c in text_chunks[:i])
        estimated_page = (
            (chunk_start // chars_per_page) + 1 if chars_per_page > 0 else 1
        )
        estimated_page = min(estimated_page, len(doc))

        chunks.append(
            Chunk(
                id=f"{doc_id}_chunk_{i}",
                text=chunk_text_content,
                metadata={
                    "filename": filename,
                    "title": title,
                    "authors": authors,
                    "page": estimated_page,
                    "chunk_index": i,
                    "chunk_type": "content",
                },
            )
        )

    doc.close()

    return Document(
        id=doc_id,
        filename=filename,
        title=title,
        title_variants=title_variants,
        authors=authors,
        abstract=abstract,
        page_count=len(pages_text),
        chunks=chunks,
    )


def load_all_pdfs(papers_dir: Path | None = None) -> list[Document]:
    """Load all PDFs from the papers directory."""
    settings = get_settings()
    directory = papers_dir or settings.papers_dir

    if not directory.exists():
        raise PDFError(
            message=f"Papers directory not found: {directory}",
            details="Create the directory or set PAPERS_DIR environment variable.",
        )

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        raise PDFError(
            message=f"No PDF files found in {directory}",
            details="Add PDF files to the papers directory.",
        )

    documents: list[Document] = []
    for pdf_path in pdf_files:
        try:
            doc = load_pdf(pdf_path)
            documents.append(doc)
        except PDFError:
            # Log warning but continue with other files
            continue

    return documents
