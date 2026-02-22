"""Unified document loading interface."""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS


def load_documents(
    source: str | Path,
    *,
    web_headers: dict[str, str] | None = None,
) -> list[Document]:
    """Load documents from a file path or URL.

    Supports:
        - PDF files (.pdf)
        - Web pages (http:// or https:// URLs)

    Args:
        source: File path (str or Path) or URL to load from.
        web_headers: Optional HTTP headers for web requests (e.g. User-Agent).
            For sec.gov, compliant headers are used by default.

    Returns:
        List of LangChain Document objects with page_content and metadata.

    Raises:
        ValueError: If source type is unsupported.
        FileNotFoundError: If the file path does not exist.
    """
    source_str = str(source).strip()
    if not source_str:
        raise ValueError("Source cannot be empty.")

    source_lower = source_str.lower()

    if source_lower.startswith(("http://", "https://")):
        logger.info("Loading from URL: %s", source_str[:80])
        loader_kwargs: dict = {"web_path": source}
        loader = WebBaseLoader(**loader_kwargs)
        docs = loader.load()
        logger.info("Loaded %d document(s) from URL", len(docs))
        return docs

    path = Path(source)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported source type: {suffix or repr(source_str[:50])}. "
            f"Supported: .pdf or http(s) URLs."
        )

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    logger.info("Loading %s: %s", suffix, path)

    loader = PyPDFLoader(file_path=path)
    docs = loader.load()

    logger.info("Loaded %d document(s) from %s", len(docs), path.name)
    return docs
