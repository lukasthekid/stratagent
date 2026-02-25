"""Analysis endpoints: async job submission, sync analysis, and status."""

import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from agents.crew import StratAgentCrew
from agents.schemas import StrategicBrief
from api.schemas import (
    AnalysisRequest,
    JobResponse,
    JobStatus,
    JobStatusResponse,
)
from api.worker.runner import job_store, run_analysis

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post(
    "/analyze-test",
    response_model=StrategicBrief,
    summary="Run synchronous analysis",
    description="Runs full multi-agent analysis and waits for the result. "
    "Returns a StrategicBrief with executive summary, SWOT, risks, recommendations, and caveats. "
    "Use POST /analysis/analyze for async (non-blocking) analysis.",
)
def analyse(request: AnalysisRequest) -> StrategicBrief:
    """Run full agent analysis synchronously. Blocks until complete."""
    try:
        crew = StratAgentCrew()
        return crew.run(company=request.company, question=request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/analyze",
    response_model=JobResponse,
    summary="Start async analysis job",
    description="Starts analysis in the background. Poll GET /analysis/jobs/{job_id} for status and result.",
)
async def analyze(request: AnalysisRequest, background_tasks: BackgroundTasks) -> JobResponse:
    """Start an async analysis job and return job_id immediately."""
    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "status": "pending",
        "result": None,
        "error": None,
        "current_phase": None,
        "current_agent": None,
        "current_tool": None,
        "progress_message": None,
    }
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
        current_phase=j.get("current_phase"),
        current_agent=j.get("current_agent"),
        current_tool=j.get("current_tool"),
        progress_message=j.get("progress_message"),
    )


@router.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
