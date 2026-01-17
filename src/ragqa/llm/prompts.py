"""Prompt templates for RAG Q&A."""

SPECIFIC_QUERY_PROMPT = """You are a research assistant. Answer ONLY using the provided context.

STRICT RULES:
- Sources are listed in order of relevance - PRIORITIZE the first source
- ONLY cite filenames that appear in [Source: filename.pdf] tags
- Pay attention to [Title: ...] - use sources whose titles match the question
- If information is not in the context, say "Not found in provided documents"
- Keep response concise (under 150 words)
- Cite using format: [filename.pdf]

Context:
{context}

Question: {question}

Answer:"""


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


def format_chunk_with_source(filename: str, title: str, page: int, text: str) -> str:
    """Format a chunk with source attribution including title."""
    return f"""[Source: {filename}]
[Title: {title}]
[Page: {page}]
{text}
---"""


def build_context(chunks: list[tuple[str, str, int, str]]) -> str:
    """Build context string from chunks (filename, title, page, text)."""
    return "\n\n".join(
        format_chunk_with_source(filename, title, page, text)
        for filename, title, page, text in chunks
    )
