"""RAG Q&A - Research Paper Q&A System."""

import logging
import os
import sys
import warnings

import structlog

# Suppress onnxruntime warnings about GPU (must be set before import)
os.environ["ORT_LOG_LEVEL"] = "ERROR"
os.environ["ONNXRUNTIME_DISABLE_WARNINGS"] = "1"

# Disable ChromaDB telemetry completely
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "True"

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")
warnings.filterwarnings("ignore", message=".*device_discovery.*")

# Suppress chromadb/posthog logging
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("posthog").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)


def configure_logging(json_format: bool = False, level: str = "INFO") -> None:
    """Configure structlog for the application.

    Args:
        json_format: If True, output logs as JSON. If False, use console format.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    # Shared processors
    shared_processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        # JSON format for production
        processors = [
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = [
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ]

    structlog.configure(
        processors=processors,  # type: ignore[arg-type]
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper()),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger for a module.

    Args:
        name: Module name, typically __name__.

    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)  # type: ignore[no-any-return]


# Default configuration (console format, WARNING level)
# Can be reconfigured by calling configure_logging()
configure_logging(json_format=False, level="WARNING")

__version__ = "0.1.0"
