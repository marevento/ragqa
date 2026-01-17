"""Typer CLI application for RAG Q&A."""

# Suppress warnings before any imports (must be at top)
import io
import logging
import os
import readline  # noqa: F401 - enables arrow key history for input()
import sys

os.environ["ORT_LOG_LEVEL"] = "ERROR"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "True"

# Suppress stderr during chromadb import (onnxruntime C++ warnings)
# Use file descriptor redirect to capture C++ level warnings
_stderr = sys.stderr
sys.stderr = io.StringIO()
_stderr_fd = os.dup(2)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)

logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("posthog").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# Monkey-patch posthog capture to suppress errors
try:
    import posthog

    posthog.capture = lambda *args, **kwargs: None
    posthog.disabled = True
except ImportError:
    pass

# Import chromadb-related modules while stderr is suppressed
from ragqa.retrieval.vectorstore import VectorStore  # noqa: E402

# Restore stderr after chromadb import
os.dup2(_stderr_fd, 2)
os.close(_stderr_fd)
os.close(_devnull)
sys.stderr = _stderr

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import typer  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.progress import Progress, SpinnerColumn, TextColumn  # noqa: E402

from ragqa.cli.banner import get_banner  # noqa: E402
from ragqa.cli.display import (  # noqa: E402
    console,
    print_banner,
    print_documents_table,
    print_error,
    print_info,
    print_response,
    print_success,
    print_warning,
)
from ragqa.config import get_settings  # noqa: E402
from ragqa.core.pdf_loader import load_all_pdfs  # noqa: E402
from ragqa.core.rag_chain import RAGChain, RAGResponse  # noqa: E402
from ragqa.exceptions import RAGError  # noqa: E402
from ragqa.llm.client import check_ollama_available  # noqa: E402
from ragqa.retrieval.bm25_index import BM25Index  # noqa: E402

app = typer.Typer(
    name="ragqa",
    help="RAG Q&A - Research Paper Q&A System",
    no_args_is_help=True,
)

# Global debug flag
_debug = False

def load_golden_tests() -> list[dict[str, str | list[str]]]:
    """Load golden test cases from external JSON file."""
    # Try multiple locations for the test file
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "tests" / "golden" / "test_cases.json",
        Path.cwd() / "tests" / "golden" / "test_cases.json",
    ]
    for path in possible_paths:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    # Fallback to empty list if file not found
    return []

# Preset questions for chat mode
PRESET_QUESTIONS = [
    "Describe the documents",
    "What is ToolMem?",
    "How can I build a LLM system that handles complex tasks?",
]


def check_prerequisites() -> bool:
    """Check if Ollama is available."""
    if not check_ollama_available():
        print_error(
            RAGError(
                message="Cannot connect to Ollama. Is it running?",
                details="Run 'ollama serve' to start Ollama.",
            ),
            _debug,
        )
        return False
    return True


def stream_response(chain: RAGChain, question: str) -> RAGResponse | None:
    """Stream a RAG response with spinner and live markdown display.

    Returns the RAGResponse if successful, None if the generator returned early.
    """
    with console.status("[dim]Thinking...[/dim]", spinner="dots") as status:
        gen = chain.ask(question, stream=True)
        if isinstance(gen, RAGResponse):
            status.stop()
            print_response(gen)
            return gen

        # Get first token while showing spinner
        full_text = ""
        response: RAGResponse | None = None
        try:
            first_token = next(gen)
            full_text = first_token
        except StopIteration as e:
            response = e.value
            status.stop()
            if response:
                print_response(response)
            return response

        status.stop()

    # Continue streaming remaining tokens
    with Live(console=console, refresh_per_second=10) as live:
        live.update(Markdown(full_text))
        try:
            while True:
                token = next(gen)
                full_text += token
                live.update(Markdown(full_text))
        except StopIteration as e:
            response = e.value

    # Print sources from the RAGResponse
    if response and full_text:
        console.print()
        console.print("-" * 40, style="dim")
        console.print(f"Confidence: {response.confidence}%", style="dim")
        console.print()
        if response.sources:
            console.print("References:", style="bold")
            for i, source in enumerate(response.sources, 1):
                title = source.get("title", "Untitled")
                authors = source.get("authors", "Unknown")
                filename = source.get("filename", "")
                console.print(f'[{i}] "{title}"', style="cyan")
                console.print(f"    {authors} | {filename}", style="dim")

    return response


@app.callback()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="Show detailed errors"),
) -> None:
    """RAG Q&A - Research Paper Q&A System."""
    global _debug
    _debug = debug


@app.command()
def index(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-index all documents"
    ),
) -> None:
    """Build or rebuild the document index."""
    if not check_prerequisites():
        raise typer.Exit(1)

    settings = get_settings()
    vectorstore = VectorStore()
    bm25_index = BM25Index()

    if vectorstore.is_indexed() and not force:
        print_info(
            f"Index already exists with {vectorstore.document_count()} documents. "
            "Use --force to rebuild."
        )
        return

    if force:
        print_info("Clearing existing index...")
        vectorstore.clear()
        bm25_index.clear()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading PDFs...", total=None)

            documents = load_all_pdfs(settings.papers_dir)
            progress.update(task, description=f"Found {len(documents)} documents")

            for i, doc in enumerate(documents):
                progress.update(
                    task,
                    description=f"Indexing {doc.filename}... ({i + 1}/{len(documents)})",
                )
                vectorstore.add_document(doc)

            progress.update(task, description="Building BM25 index...")
            bm25_index.build_from_documents(documents)

            progress.update(task, description="Done!")

        print_success(
            f"Indexed {vectorstore.document_count()} documents "
            f"({vectorstore.chunk_count()} chunks)"
        )

    except RAGError as e:
        print_error(e, _debug)
        raise typer.Exit(1) from e
    except Exception as e:
        print_error(RAGError(message="Indexing failed", details=str(e)), _debug)
        raise typer.Exit(1) from e


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    no_stream: bool = typer.Option(
        False, "--no-stream", help="Disable streaming output"
    ),
) -> None:
    """Ask a question about the research papers."""
    if not check_prerequisites():
        raise typer.Exit(1)

    if not question.strip():
        print_warning("Please enter a question")
        raise typer.Exit(1)

    try:
        vectorstore = VectorStore()
        if not vectorstore.is_indexed():
            print_error(
                RAGError(
                    message="No documents indexed",
                    details="Run 'ragqa index' first to build the index.",
                ),
                _debug,
            )
            raise typer.Exit(1)

        chain = RAGChain(vectorstore)

        if no_stream or json_output:
            result = chain.ask(question, stream=False)
            if isinstance(result, RAGResponse):
                if json_output:
                    output = {
                        "answer": result.answer,
                        "sources": result.sources,
                        "confidence": result.confidence,
                        "query_type": result.query_type,
                    }
                    print(json.dumps(output, indent=2))
                else:
                    print_response(result)
        else:
            console.print()
            stream_response(chain, question)

    except RAGError as e:
        print_error(e, _debug)
        raise typer.Exit(1) from e
    except Exception as e:
        print_error(RAGError(message="Query failed", details=str(e)), _debug)
        raise typer.Exit(1) from e


@app.command()
def chat() -> None:
    """Interactive chat mode with preset questions."""
    if not check_prerequisites():
        raise typer.Exit(1)

    try:
        vectorstore = VectorStore()
        if not vectorstore.is_indexed():
            print_error(
                RAGError(
                    message="No documents indexed",
                    details="Run 'ragqa index' first to build the index.",
                ),
                _debug,
            )
            raise typer.Exit(1)

        # Print banner
        banner = get_banner(vectorstore.document_count())
        print_banner(banner)

        chain = RAGChain(vectorstore)

        while True:
            console.print()
            console.print("Choose a question or type your own:", style="bold")
            for i, q in enumerate(PRESET_QUESTIONS, 1):
                console.print(f"  [{i}] {q}", style="cyan")
            console.print("  [Q] Quit", style="red")
            console.print()

            try:
                # Use input() instead of console.input() for readline history support
                console.print("[bold]> [/bold]", end="")
                user_input = input().strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() == "q":
                console.print("Goodbye!")
                break

            # Check if user selected a preset
            if user_input.isdigit():
                idx = int(user_input) - 1
                if 0 <= idx < len(PRESET_QUESTIONS):
                    question = PRESET_QUESTIONS[idx]
                else:
                    print_warning("Invalid selection")
                    continue
            else:
                question = user_input

            console.print()
            console.print(f"[dim]Question: {question}[/dim]")
            stream_response(chain, question)

    except RAGError as e:
        print_error(e, _debug)
        raise typer.Exit(1) from e
    except Exception as e:
        print_error(RAGError(message="Chat failed", details=str(e)), _debug)
        raise typer.Exit(1) from e


@app.command(name="list-docs")
def list_docs() -> None:
    """List all indexed documents."""
    try:
        vectorstore = VectorStore()
        if not vectorstore.is_indexed():
            print_info("No documents indexed. Run 'ragqa index' first.")
            return

        documents = vectorstore.get_all_documents()
        print_documents_table(documents)

    except RAGError as e:
        print_error(e, _debug)
        raise typer.Exit(1) from e


@app.command()
def config() -> None:
    """Display current configuration."""
    settings = get_settings()

    console.print()
    console.print("Current Configuration:", style="bold")
    console.print(f"  Ollama URL: {settings.ollama_base_url}")
    console.print(f"  LLM Model: {settings.ollama_model}")
    console.print(f"  Embedding Model: {settings.ollama_embed_model}")
    console.print(f"  Papers Directory: {settings.papers_dir}")
    console.print(f"  ChromaDB Directory: {settings.chroma_persist_dir}")
    console.print(f"  Chunk Size: {settings.chunk_size}")
    console.print(f"  Chunk Overlap: {settings.chunk_overlap}")
    console.print(f"  Top-K Results: {settings.retrieval_top_k}")
    console.print(f"  Debug Mode: {settings.debug}")
    console.print()


@app.command()
def test(
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
) -> None:
    """Run golden file tests."""
    if not check_prerequisites():
        raise typer.Exit(1)

    try:
        vectorstore = VectorStore()
        if not vectorstore.is_indexed():
            print_error(
                RAGError(
                    message="No documents indexed",
                    details="Run 'ragqa index' first to build the index.",
                ),
                _debug,
            )
            raise typer.Exit(1)

        chain = RAGChain(vectorstore)
        doc_count = vectorstore.document_count()
        results = []
        golden_tests = load_golden_tests()

        if not golden_tests:
            print_warning("No golden test cases found in tests/golden/test_cases.json")
            raise typer.Exit(1)

        if not json_output:
            console.print()
            console.print("Running golden file tests...", style="bold")
            console.print()

        for i, test_case in enumerate(golden_tests, 1):
            query = str(test_case["query"])
            expected = test_case["expected_sources"]
            keywords_raw = test_case["keywords"]
            keywords = list(keywords_raw) if isinstance(keywords_raw, list) else []

            if not json_output:
                console.print(f'[{i}/{len(golden_tests)}] "{query}"')

            result = chain.ask(query, stream=False)
            if not isinstance(result, RAGResponse):
                continue

            actual_sources = [str(s.get("filename", "")) for s in result.sources]

            # Check sources
            if expected == "all":
                sources_ok = len(actual_sources) >= doc_count - 2
            else:
                expected_list = list(expected) if isinstance(expected, list) else [str(expected)]
                sources_ok = any(
                    exp in actual_sources or any(exp in a for a in actual_sources)
                    for exp in expected_list
                )

            # Check keywords in answer
            answer_lower = result.answer.lower()
            keywords_found = [str(kw) for kw in keywords if str(kw).lower() in answer_lower]
            keywords_ok = (
                len(keywords_found) >= len(keywords) // 2 if keywords else True
            )

            passed = sources_ok and keywords_ok

            test_result = {
                "query": query,
                "expected_sources": expected,
                "actual_sources": actual_sources,
                "passed": passed,
                "checks": {
                    "sources_ok": sources_ok,
                    "keywords_ok": keywords_ok,
                    "keywords_found": keywords_found,
                },
            }
            results.append(test_result)

            if not json_output:
                if sources_ok:
                    console.print(
                        f"  [green]✓[/green] Sources: {', '.join(actual_sources[:3])}"
                    )
                else:
                    console.print(
                        f"  [red]✗[/red] Sources: expected {expected}, got {actual_sources}"
                    )

                if keywords:
                    if keywords_ok:
                        console.print(
                            f"  [green]✓[/green] Keywords: {', '.join(keywords_found)}"
                        )
                    else:
                        console.print(f"  [red]✗[/red] Keywords: expected {keywords}")
                console.print()

        passed_count = sum(1 for r in results if r["passed"])
        total = len(results)

        if json_output:
            output = {
                "tests": results,
                "summary": {
                    "passed": passed_count,
                    "failed": total - passed_count,
                    "total": total,
                },
            }
            print(json.dumps(output, indent=2))
        else:
            console.print("-" * 40)
            style = "green" if passed_count == total else "red"
            console.print(f"Results: [{style}]{passed_count}/{total} passed[/{style}]")

        if passed_count < total:
            raise typer.Exit(1)

    except RAGError as e:
        print_error(e, _debug)
        raise typer.Exit(1) from e
    except typer.Exit:
        raise
    except Exception as e:
        print_error(RAGError(message="Test failed", details=str(e)), _debug)
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
