"""Rich output formatting for CLI."""

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ragqa.core.rag_chain import RAGResponse
from ragqa.exceptions import RAGError

console = Console()


def print_banner(banner: str) -> None:
    """Print the application banner."""
    console.print(banner, style="bold cyan")


def print_response(response: RAGResponse) -> None:
    """Print a RAG response with sources."""
    # Print answer
    console.print()
    console.print(Markdown(response.answer))
    console.print()

    # Print separator and confidence
    console.print("-" * 40, style="dim")
    console.print(f"Confidence: {response.confidence}%", style="dim")
    console.print()

    # Print references
    if response.sources:
        console.print("References:", style="bold")
        for i, source in enumerate(response.sources, 1):
            title = source.get("title", "Untitled")
            authors = source.get("authors", "Unknown")
            filename = source.get("filename", "")
            page = source.get("page", "")

            console.print(f'[{i}] "{title}"', style="cyan")
            if page:
                console.print(f"    {authors} | {filename}, page {page}", style="dim")
            else:
                console.print(f"    {authors} | {filename}", style="dim")


def print_streaming_start() -> None:
    """Print indicator that streaming is starting."""
    console.print()


def print_error(error: RAGError, debug: bool = False) -> None:
    """Print an error message."""
    if debug and error.details:
        console.print(
            Panel(
                f"[red]Error:[/red] {error.message}\n\n"
                f"[dim]Details:[/dim] {error.details}",
                title="Error",
                border_style="red",
            )
        )
    else:
        console.print(
            Panel(
                f"[red]Error:[/red] {error.message}\n\n"
                "[dim]Run with --debug for details[/dim]",
                title="Error",
                border_style="red",
            )
        )


def print_documents_table(documents: list[dict[str, object]]) -> None:
    """Print a table of indexed documents."""
    table = Table(title="Indexed Documents")
    table.add_column("Filename", style="cyan")
    table.add_column("Title", style="green")
    table.add_column("Authors", style="yellow")
    table.add_column("Pages", justify="right")

    for doc in documents:
        meta = doc.get("metadata", {})
        if isinstance(meta, dict):
            filename = str(meta.get("filename", ""))
            title = str(meta.get("title", "Untitled"))[:50]
            if len(str(meta.get("title", ""))) > 50:
                title += "..."
            authors = str(meta.get("authors", "Unknown"))
            pages = str(meta.get("page_count", "?"))
            table.add_row(filename, title, authors, pages)

    console.print(table)


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")
