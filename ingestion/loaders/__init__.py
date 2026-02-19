"""Document loaders for PDF, CSV, and web sources."""

from ingestion.loaders.csv_loader import CSVLoader
from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.web_loader import WebLoader

__all__ = ["PDFLoader", "CSVLoader", "WebLoader"]
