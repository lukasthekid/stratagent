"""Standardized document model for ingestion."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata attached to ingested documents."""

    source: str = Field(..., description="File path, URL, or identifier of the source")
    date: datetime | None = Field(default=None, description="Document date or ingestion date")
    company_name: str | None = Field(default=None, description="Company or organization name")
    extra: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Document(BaseModel):
    """Standardized document output from all loaders."""

    content: str = Field(..., description="Extracted text or structured content")
    metadata: DocumentMetadata = Field(..., description="Source and contextual metadata")
