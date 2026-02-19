"""Ingestion module for document processing and indexing."""

from ingestion.load import load_documents
from ingestion.loaders import CSVLoader, PDFLoader, WebLoader
from ingestion.models import Document, DocumentMetadata

__all__ = [
    "Document",
    "DocumentMetadata",
    "load_documents",
    "PDFLoader",
    "CSVLoader",
    "WebLoader",
]
