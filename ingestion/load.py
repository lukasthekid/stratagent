"""Unified document loading interface."""

from pathlib import Path
from typing import Any

from ingestion.loaders.csv_loader import CSVLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.web_loader import WebLoader
from ingestion.models import Document

CSV_EXTENSIONS = {".csv"}
PDF_EXTENSIONS = {".pdf"}


def load_documents(
    source: str | Path,
    *,
    company_name: str | None = None,
    source_date: Any = None,
    csv_schema: type | None = None,
    csv_encoding: str = "utf-8",
) -> list[Document]:
    """Load documents from a file path or URL.

    Dispatches to the appropriate loader based on source type:
    - .pdf -> PDFLoader
    - .csv -> CSVLoader
    - http(s):// -> WebLoader

    Args:
        source: File path or URL.
        company_name: Optional company name for metadata.
        source_date: Optional document date.
        csv_schema: Optional Pydantic model for CSV validation.
        csv_encoding: CSV file encoding (default: utf-8).

    Returns:
        List of standardized Documents.

    Raises:
        ValueError: If source type is not supported.
    """
    source_str = str(source).strip().lower()

    if source_str.startswith(("http://", "https://")):
        loader = WebLoader(source, company_name=company_name, source_date=source_date)
        return loader.load()

    path = Path(source)
    suffix = path.suffix.lower()

    if suffix in PDF_EXTENSIONS:
        loader = PDFLoader(path, company_name=company_name, source_date=source_date)
        return loader.load()

    if suffix in CSV_EXTENSIONS:
        loader = CSVLoader(
            path,
            company_name=company_name,
            source_date=source_date,
            schema=csv_schema,
            encoding=csv_encoding,
        )
        return loader.load()

    raise ValueError(
        f"Unsupported source type: {suffix or source_str[:50]}. "
        f"Supported: .pdf, .csv, or http(s) URLs."
    )
