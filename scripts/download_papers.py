#!/usr/bin/env python3
"""Download research papers from arXiv.

Usage:
    python scripts/download_papers.py 2510.06042 2510.06534
    python scripts/download_papers.py --file papers.txt
    python scripts/download_papers.py --file papers.txt --output-dir ./my_papers
"""

import argparse
import re
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}.pdf"
ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")


def normalize_arxiv_id(paper_id: str) -> str:
    """Normalize arXiv ID by stripping URL prefixes and extracting the ID."""
    paper_id = paper_id.strip()

    # Handle full URLs — validate the hostname to avoid substring spoofing
    # (e.g., "evil.com/arxiv.org" must not be treated as an arXiv URL)
    if "://" in paper_id:
        parsed = urlparse(paper_id)
        hostname = parsed.hostname.lower() if parsed.hostname else None
        if hostname == "arxiv.org" or hostname.endswith(".arxiv.org"):
            match = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", parsed.path)
            if match:
                return match.group(0)

    return paper_id


def validate_arxiv_id(paper_id: str) -> bool:
    """Validate arXiv paper ID format (YYMM.NNNNN or YYMM.NNNNNvN)."""
    return bool(ARXIV_ID_PATTERN.match(paper_id))


def download_paper(paper_id: str, output_dir: Path, verbose: bool = True) -> bool:
    """Download a paper from arXiv.

    Args:
        paper_id: arXiv paper ID (e.g., "2510.06042" or "2510.06042v1")
        output_dir: Directory to save the PDF
        verbose: Print progress messages

    Returns:
        True if download succeeded, False otherwise
    """
    paper_id = normalize_arxiv_id(paper_id)

    if not validate_arxiv_id(paper_id):
        print(f"Invalid arXiv ID format: {paper_id}", file=sys.stderr)
        return False

    # Add v1 if no version specified
    if not re.search(r"v\d+$", paper_id):
        paper_id = f"{paper_id}v1"

    output_path = output_dir / f"{paper_id}.pdf"

    if output_path.exists():
        if verbose:
            try:
                display_path = output_path.relative_to(Path.cwd())
            except ValueError:
                display_path = output_path
            print(f"Already exists: {display_path}")
        return True

    url = ARXIV_PDF_URL.format(paper_id=paper_id)

    if verbose:
        print(f"Downloading {paper_id}...", end=" ", flush=True)

    try:
        # Add User-Agent header to avoid 403 errors
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (RAGacademic paper downloader)"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()

            # Verify we got a PDF
            if not content.startswith(b"%PDF"):
                print(f"ERROR: Response is not a PDF", file=sys.stderr)
                return False

            output_path.write_bytes(content)

        if verbose:
            print(f"OK ({len(content) / 1024:.1f} KB)")
        return True

    except urllib.error.HTTPError as e:
        print(f"ERROR: HTTP {e.code} - {e.reason}", file=sys.stderr)
        return False
    except urllib.error.URLError as e:
        print(f"ERROR: {e.reason}", file=sys.stderr)
        return False
    except TimeoutError:
        print("ERROR: Request timed out", file=sys.stderr)
        return False


def load_paper_ids_from_file(filepath: Path) -> list[str]:
    """Load paper IDs from a text file (one per line, comments with #)."""
    paper_ids = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                paper_ids.append(line)
    return paper_ids


def main():
    parser = argparse.ArgumentParser(
        description="Download research papers from arXiv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s 2510.06042 2510.06534
    %(prog)s --file papers.txt
    %(prog)s 2510.06042v2 --output-dir ./papers
    %(prog)s https://arxiv.org/abs/2510.06042
        """
    )
    parser.add_argument(
        "paper_ids",
        nargs="*",
        help="arXiv paper IDs to download (e.g., 2510.06042 or 2510.06042v1)"
    )
    parser.add_argument(
        "-f", "--file",
        type=Path,
        help="File containing paper IDs (one per line)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "research_papers",
        help="Output directory for PDFs (default: ./research_papers)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    # Collect paper IDs from arguments and/or file
    paper_ids = list(args.paper_ids)
    if args.file:
        if not args.file.exists():
            print(f"File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        paper_ids.extend(load_paper_ids_from_file(args.file))

    if not paper_ids:
        parser.print_help()
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Download papers
    success_count = 0
    for paper_id in paper_ids:
        if download_paper(paper_id, args.output_dir, verbose=not args.quiet):
            success_count += 1

    # Summary
    if not args.quiet:
        # Show relative path if within cwd
        try:
            display_path = args.output_dir.resolve().relative_to(Path.cwd())
        except ValueError:
            display_path = args.output_dir
        print(f"\nDownloaded {success_count}/{len(paper_ids)} papers to {display_path}")

    sys.exit(0 if success_count == len(paper_ids) else 1)


if __name__ == "__main__":
    main()
