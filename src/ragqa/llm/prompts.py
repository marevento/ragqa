"""Prompt templates for RAG Q&A."""

SPECIFIC_QUERY_PROMPT = """You are a research assistant. Answer ONLY using the provided context.

STRICT RULES:
- ONLY cite filenames that appear in [Source: filename.pdf] tags below
- NEVER invent paper titles, author names, or statistics
- If information is not in the context, say "Not found in provided documents"
- Keep response under 200 words
- Use format: "According to [filename.pdf], ..."

Context:
{context}

Question: {question}

Answer (cite sources exactly as shown):"""


ALL_DOCS_PROMPT = """Group these research papers by topic and provide a brief overview.

Papers:
{titles}

Format your response as:
## [Topic Name]
- **paper_filename.pdf**: Brief description based on title

Group related papers together. Identify 3-5 main themes. Keep descriptions to one sentence each."""


SINGLE_DOC_PROMPT = """Summarize this research paper in 2-3 sentences based on the provided context.

Context:
{context}

Paper: {filename}

Summary:"""


def format_chunk_with_source(filename: str, page: int, text: str) -> str:
    """Format a chunk with source attribution."""
    return f"""[Source: {filename}]
[Page: {page}]
{text}
---"""


def build_context(chunks: list[tuple[str, int, str]]) -> str:
    """Build context string from chunks (filename, page, text)."""
    return "\n\n".join(
        format_chunk_with_source(filename, page, text)
        for filename, page, text in chunks
    )
