"""API request and response schemas."""

from pydantic import BaseModel, Field


class AnalyseRequest(BaseModel):
    """Request body for POST /analyse."""

    company: str = Field(..., description="Company name to analyse")
    query: str = Field(..., description="Strategic question to answer about the company")


class IngestUrlRequest(BaseModel):
    """Request body for POST /ingest/url."""

    url: str = Field(..., description="URL to load (http:// or https://)")


class FileIngestResult(BaseModel):
    """Result for a single ingested file."""

    filename: str
    documents: int
    chunks: int


class IngestResponse(BaseModel):
    """Response for ingest endpoints."""

    chunk_ids: list[str] = Field(..., description="Pinecone IDs of upserted chunks")
    chunk_count: int = Field(..., description="Total number of chunks upserted")
    files: list[FileIngestResult] = Field(
        default_factory=list,
        description="Per-file breakdown (upload only)",
    )
