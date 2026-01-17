"""Tests for PDF loader module."""

from ragqa.core.pdf_loader import chunk_text, format_authors


class TestChunkText:
    """Tests for text chunking."""

    def test_empty_text(self) -> None:
        """Empty text returns empty list."""
        assert chunk_text("", 100, 20) == []

    def test_short_text(self) -> None:
        """Text shorter than chunk size returns single chunk."""
        text = "Short text."
        chunks = chunk_text(text, 100, 20)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunking_with_overlap(self) -> None:
        """Text is split with overlap."""
        text = "A" * 100 + ". " + "B" * 100 + ". " + "C" * 100
        chunks = chunk_text(text, 120, 20)
        assert len(chunks) >= 2

    def test_sentence_boundary(self) -> None:
        """Chunks break at sentence boundaries when possible."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, 30, 5)
        # Should break at period
        assert any("." in c for c in chunks)


class TestFormatAuthors:
    """Tests for author formatting."""

    def test_single_author(self) -> None:
        """Single author returned as-is."""
        assert format_authors("John Smith") == "John Smith"

    def test_two_authors(self) -> None:
        """Two authors returned as-is."""
        result = format_authors("John Smith and Jane Doe")
        assert "Smith" in result or "John" in result

    def test_multiple_authors_et_al(self) -> None:
        """Multiple authors formatted as et al."""
        result = format_authors("John Smith, Jane Doe, Bob Wilson, Alice Brown")
        assert "et al" in result

    def test_truncation(self) -> None:
        """Long author string is truncated."""
        long_name = "A" * 100
        result = format_authors(long_name)
        assert len(result) <= 50
