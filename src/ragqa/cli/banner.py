"""ASCII art banner for RAG Q&A."""

from ragqa.config import get_settings


def get_banner(doc_count: int = 0) -> str:
    """Generate the application banner."""
    settings = get_settings()
    model = settings.ollama_model

    doc_info = f"{doc_count} papers" if doc_count > 0 else "no papers indexed"

    return f"""
  +-----------------------------------------+
  |                                         |
  |    +---+      RAG Q&A                   |
  |   +-+-+|      Research Paper Q&A        |
  |  +-+-+||      ---------------------     |
  |  |###|`|      {doc_info} | {model:<12} |
  |  |###+-+                                |
  |  +---+                                  |
  |                                         |
  +-----------------------------------------+
"""
