# RAG Q&A - Research Paper Q&A System

A CLI application for answering questions about research papers using RAG (Retrieval-Augmented Generation) with Ollama as the LLM backend.

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                         RAG Q&A CLI                           │
│  ┌───────┐  ┌─────────┐  ┌────────┐  ┌──────────────────────┐ │
│  │ index │  │   ask   │  │  chat  │  │         test         │ │
│  └───┬───┘  └────┬────┘  └───┬────┘  └──────────┬───────────┘ │
└──────┼──────────┼───────────┼───────────────────┼─────────────┘
       │          │           │                   │
       ▼          ▼           ▼                   ▼
┌───────────────────────────────────────────────────────────────┐
│                          RAG Chain                            │
│  ┌─────────────────┐  ┌────────────┐  ┌───────────────────┐   │
│  │ Query Classifier│─▶│  Retriever │─▶│   LLM Generator   │   │
│  └─────────────────┘  └─────┬──────┘  └───────────────────┘   │
└─────────────────────────────┼─────────────────────────────────┘
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
┌─────────────┐      ┌───────────────┐      ┌───────────────┐
│ PDF Loader  │      │ Vector Store  │      │  Ollama API   │
│  (PyMuPDF)  │      │  (ChromaDB)   │      │ (LLM + Embed) │
└─────────────┘      └───────────────┘      └───────────────┘
```

## Features

- **PDF Processing**: Extracts text, titles, authors, and abstracts from research papers
- **Hybrid Search**: Combines BM25 keyword search + semantic embeddings with RRF fusion
- **Title Boosting**: Prioritizes documents whose titles match the query
- **Document Filtering**: Score-based filtering to show only relevant sources
- **Query Classification**: Routes queries to appropriate handlers (all-docs, single-doc, specific)
- **Streaming Output**: Real-time token display for LLM responses
- **Interactive Chat**: Preset questions with arrow key history navigation
- **Golden Tests**: Built-in verification tests
- **Async Support**: Full async/await API for integration with async frameworks
- **Structured Logging**: Production-ready logging with structlog
- **Protocol-based DI**: Swappable components via Python protocols
- **Performance Caching**: LRU cache for embeddings and title similarity scores

## Installation

### Prerequisites

1. **Python 3.10+**
2. **Ollama** running locally with required models:
   ```bash
   ollama serve
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```

### Setup

```bash
# Clone and enter directory
cd ragqa

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install poetry
poetry install

# Copy environment template
cp .env.example .env

# Download research papers from arXiv
python scripts/download_papers.py --file papers.txt
```

## Usage

### Index Documents

```bash
# Build the index (first time)
ragqa index

# Force rebuild
ragqa index --force
```

### Ask Questions

```bash
# Single question
ragqa ask "What is ToolMem?"

# JSON output
ragqa ask --json "What is ToolMem?"

# No streaming
ragqa ask --no-stream "What is ToolMem?"
```

### Interactive Chat

```bash
ragqa chat
```

### List Documents

```bash
ragqa list-docs
```

### Run Tests

```bash
# Golden file tests
ragqa test

# JSON output for CI
ragqa test --json
```

### Configuration

```bash
ragqa config
```

## Configuration

Environment variables (in `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen2.5:7b` | LLM model for generation |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `PAPERS_DIR` | `./research_papers` | PDF papers directory |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DEBUG` | `false` | Enable debug mode |

## Design Decisions

### Why ChromaDB?
- Embedded database, no external infrastructure
- Persistent storage with automatic loading
- Native cosine similarity support

### Why PyMuPDF?
- Fast PDF parsing (faster than pypdf)
- Good text extraction with layout preservation
- Handles multi-column layouts

### Why Hybrid Search?
- BM25 handles exact keyword matches (e.g., "ToolMem")
- Semantic search captures conceptual similarity
- Reciprocal Rank Fusion (RRF) merges results effectively
- Title boosting improves precision for document-specific queries
- Document-level score filtering reduces irrelevant sources

### Query Classification
- Keyword heuristics for predictable routing
- Handles three query types:
  - `all_docs`: Overview of all papers
  - `single_doc`: Summary of specific paper
  - `specific`: Factual/how-to questions

### Anti-Hallucination Measures
- Low temperature (0.1) for factual responses
- Explicit source tags in prompts
- Instructions to only cite provided sources
- Response word limits

## Project Structure

```
ragqa/
├── src/ragqa/
│   ├── __init__.py        # Logging configuration
│   ├── config.py          # Settings (pydantic-settings)
│   ├── exceptions.py      # Custom exceptions
│   ├── protocols.py       # Protocol interfaces for DI
│   ├── core/
│   │   ├── models.py      # Data models (Chunk, Document)
│   │   ├── pdf_loader.py  # PDF parsing
│   │   └── rag_chain.py   # RAG orchestration (sync + async)
│   ├── retrieval/
│   │   ├── embeddings.py  # Ollama embeddings (sync + async)
│   │   ├── vectorstore.py # ChromaDB wrapper
│   │   ├── bm25_index.py  # BM25 keyword search
│   │   ├── retriever.py   # Hybrid search + RRF
│   │   └── query_classifier.py
│   ├── llm/
│   │   ├── client.py      # Ollama client (sync + async)
│   │   └── prompts.py     # Prompt templates
│   └── cli/
│       ├── app.py         # Typer commands
│       ├── display.py     # Rich output
│       └── banner.py      # ASCII banner
└── tests/
    ├── conftest.py        # Shared fixtures
    ├── golden/            # Golden test cases
    ├── core/              # Core module tests
    ├── retrieval/         # Retrieval tests
    └── llm/               # LLM client tests
```

## Development

```bash
# Run tests
poetry run pytest

# Type checking
poetry run mypy src/

# Linting
poetry run ruff check src/ tests/

# Format
poetry run ruff format src/ tests/
```

## Possible Improvements

- **Conversation Memory**: Add context persistence in chat mode using sliding window or summarization
- **Query Classification**: Replace keyword heuristics with embedding-based classification using labeled examples
- **Retrieval Enhancement**: Add cross-encoder re-ranking after RRF fusion for improved relevance
- **Incremental Indexing**: Implement hash-based change detection to avoid full re-index on document updates
- **Expanded Golden Tests**: Add more test cases covering edge cases, multi-document queries, and failure modes

## License

MIT
