"""Query classification for routing to appropriate handlers."""

import re
from typing import Literal

QueryType = Literal["all_docs", "single_doc", "specific"]

# Patterns for all-docs summarization queries
ALL_DOCS_PATTERNS = [
    "describe the documents",
    "describe all",
    "summarize all",
    "list all",
    "overview of all",
    "each paper",
    "all papers",
    "all documents",
    "what papers",
    "what documents",
]


def classify_query(query: str) -> QueryType:
    """Classify query type to route to appropriate handler."""
    query_lower = query.lower()

    # All-docs summarization
    if any(p in query_lower for p in ALL_DOCS_PATTERNS):
        return "all_docs"

    # Single-doc summarization (mentions specific arxiv-style ID)
    single_doc_keywords = ["summarize", "summary", "about", "what is"]
    if re.search(r"2510\.\d+", query) and any(
        w in query_lower for w in single_doc_keywords
    ):
        return "single_doc"

    # Default: specific query (factual, how-to)
    return "specific"
