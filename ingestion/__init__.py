"""Ingestion module for document processing and indexing."""

from ingestion.load import load_documents
from ingestion.upsert import upsert_documents

__all__ = [
    "load_documents",
    "upsert_documents"
]


