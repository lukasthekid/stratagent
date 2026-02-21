"""Analysis endpoints: async job submission and status."""

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas import AnalysisRequest, JobResponse, JobStatus, JobStatusResponse
from api.worker.runner import job_store, run_analysis

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post(
    "/analyze",
    response_model=JobResponse,
    summary="Start async analysis job",
    description="Starts analysis in the background. Poll GET /analysis/jobs/{job_id} for status and result.",
)
async def analyze(request: AnalysisRequest, background_tasks: BackgroundTasks) -> JobResponse:
    """Start an async analysis job and return job_id immediately."""
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "pending", "result": None, "error": None}
    background_tasks.add_task(run_analysis, job_id, request.company, request.question)
    return JobResponse(
        job_id=job_id,
        status=JobStatus.pending,
        message="Analysis started",
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status and result",
    description="Returns job status. When completed, result contains the StrategicBrief. When failed, error contains the message.",
)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Get status and result of an analysis job."""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    j = job_store[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=JobStatus(j["status"]),
        result=j.get("result"),
        error=j.get("error"),
    )


@router.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
