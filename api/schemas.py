"""API request and response schemas."""

from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of an analysis job."""

    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class AnalysisRequest(BaseModel):
    """Request body for POST /analyze."""

    company: str = Field(..., description="Company name", examples=["Tesla"])
    question: str = Field(
        ...,
        description="Strategic question to answer",
        examples=["What are Tesla's biggest strategic risks in 2025?"],
    )


class JobResponse(BaseModel):
    """Response for POST /analyze (job submitted)."""

    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    """Response for GET /jobs/{job_id}."""

    job_id: str
    status: JobStatus
    result: dict | None = None
    error: str | None = None


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
