"""RAG Q&A - Research Paper Q&A System."""

import logging
import os
import warnings

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

__version__ = "0.1.0"
