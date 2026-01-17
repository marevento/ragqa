# RAG Q&A - Research Paper Q&A System

A CLI application for answering questions about research papers using RAG (Retrieval-Augmented Generation) with Ollama as the LLM backend.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           RAG Q&A CLI                                │
│  ┌─────────┐  ┌───────────────┐  ┌────────────────┐  ┌───────────┐  │
│  │  index  │  │     ask       │  │     chat       │  │   test    │  │
│  └────┬────┘  └───────┬───────┘  └───────┬────────┘  └─────┬─────┘  │
└───────┼───────────────┼──────────────────┼─────────────────┼────────┘
        │               │                  │                 │
        ▼               ▼                  ▼                 ▼
┌───────────────────────────────────────────────────────────────────┐
│                        RAG Chain                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────┐  │
│  │ Query Classifier │──│    Retriever    │──│   LLM Generator   │  │
│  └─────────────────┘  └────────┬────────┘  └───────────────────┘  │
└────────────────────────────────┼──────────────────────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  PDF Loader   │      │  Vector Store   │      │   Ollama API    │
│  (PyMuPDF)    │      │  (ChromaDB)     │      │  (LLM + Embed)  │
└───────────────┘      └─────────────────┘      └─────────────────┘
```

## Features

- **PDF Processing**: Extracts text, titles, authors, and abstracts from research papers
- **Semantic Search**: Uses ChromaDB with nomic-embed-text embeddings
- **Query Classification**: Routes queries to appropriate handlers (all-docs, single-doc, specific)
- **Streaming Output**: Real-time token display for LLM responses
- **Interactive Chat**: Preset questions with custom input support
- **Golden Tests**: Built-in verification tests

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
cd RAGacademic

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install poetry
poetry install

# Copy environment template
cp .env.example .env
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

### Why Semantic-Only Search (MVP)?
- Simpler implementation for initial version
- Sufficient for most queries with good embeddings
- Hybrid BM25+semantic planned for Phase 2

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
│   ├── config.py          # Settings (pydantic-settings)
│   ├── exceptions.py      # Custom exceptions
│   ├── core/
│   │   ├── models.py      # Data models
│   │   ├── pdf_loader.py  # PDF parsing
│   │   └── rag_chain.py   # RAG orchestration
│   ├── retrieval/
│   │   ├── embeddings.py  # Ollama embeddings
│   │   ├── vectorstore.py # ChromaDB wrapper
│   │   ├── retriever.py   # Search orchestration
│   │   └── query_classifier.py
│   ├── llm/
│   │   ├── client.py      # Ollama client
│   │   └── prompts.py     # Prompt templates
│   └── cli/
│       ├── app.py         # Typer commands
│       ├── display.py     # Rich output
│       └── banner.py      # ASCII banner
└── tests/
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

## License

MIT
