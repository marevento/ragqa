"""Tests for query classifier."""

from ragqa.retrieval.query_classifier import classify_query


class TestQueryClassifier:
    """Tests for query classification."""

    def test_all_docs_describe(self) -> None:
        """'Describe the documents' classified as all_docs."""
        assert classify_query("Describe the documents") == "all_docs"

    def test_all_docs_summarize_all(self) -> None:
        """'Summarize all papers' classified as all_docs."""
        assert classify_query("Summarize all papers") == "all_docs"

    def test_all_docs_list_all(self) -> None:
        """'List all documents' classified as all_docs."""
        assert classify_query("List all documents") == "all_docs"

    def test_specific_what_is(self) -> None:
        """'What is X?' classified as specific."""
        assert classify_query("What is ToolMem?") == "specific"

    def test_specific_how_to(self) -> None:
        """'How can I...' classified as specific."""
        assert classify_query("How can I build an LLM system?") == "specific"

    def test_single_doc_summarize(self) -> None:
        """Summarize specific paper classified as single_doc."""
        assert classify_query("Summarize paper 2510.06664") == "single_doc"

    def test_case_insensitive(self) -> None:
        """Classification is case-insensitive."""
        assert classify_query("DESCRIBE THE DOCUMENTS") == "all_docs"
        assert classify_query("describe the Documents") == "all_docs"
