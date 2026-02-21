"""Background analysis runner with in-memory job store."""

import asyncio

from loguru import logger

from agents.crew import StratAgentCrew

job_store: dict[str, dict] = {}


async def run_analysis(job_id: str, company: str, question: str) -> None:
    """Run analysis in background, updating job_store with status and result."""
    job_store[job_id]["status"] = "running"
    logger.info("Starting analysis for job_id={} company={}", job_id, company)

    try:
        crew = StratAgentCrew()
        brief = await asyncio.to_thread(
            crew.run,
            company=company,
            question=question,
        )
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["result"] = brief.model_dump()
        logger.info("Analysis completed for job_id={} company={}", job_id, company)
    except Exception as e:
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)
        logger.exception("Analysis failed for job_id={} company={}: {}", job_id, company, e)
