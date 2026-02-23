import logging
import re
import time

from crewai import Crew, Process
import mlflow
from litellm import RateLimitError
from agents.critic_agent import create_critic_agent
from agents.research_agent import create_research_agent
from agents.schemas import StrategicBrief
from agents.synthesis_agent import create_synthesis_agent
from agents.tasks import create_research_task, create_synthesis_task, create_critic_task
from config import settings

logger = logging.getLogger(__name__)

#mlflow config
mlflow.crewai.autolog()
mlflow.set_tracking_uri(settings.mlflow_uri)
mlflow.set_experiment(settings.mlflow_experiment_name)

def _parse_retry_after_seconds(error: RateLimitError) -> float | None:
    """Extract 'Please try again in X.XXs' from Groq/LiteLLM rate limit error message."""
    msg = str(error)
    match = re.search(r"Please try again in (\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    if match:
        return float(match.group(1))
    if hasattr(error, "retry_after") and error.retry_after is not None:
        return float(error.retry_after)
    return None


def _run_crew_with_retry(crew: Crew, inputs: dict):
    """Run crew.kickoff with retry on rate limit errors."""
    last_error = None
    for attempt in range(1, settings.llm_rate_limit_max_retries + 1):
        try:
            return crew.kickoff(inputs=inputs)
        except RateLimitError as e:
            last_error = e
            base_wait = _parse_retry_after_seconds(e) or settings.llm_rate_limit_default_wait_seconds
            wait = base_wait + 1.0  # Add 1s buffer to avoid immediate re-hit
            if attempt < settings.llm_rate_limit_max_retries:
                logger.warning(
                    "Rate limit hit (attempt %d/%d), waiting %.1fs before retry: %s",
                    attempt,
                    settings.llm_rate_limit_max_retries,
                    wait,
                    str(e)[:200],
                )
                time.sleep(wait)
            else:
                logger.error(
                    "Rate limit exceeded after %d attempts. Last error: %s",
                    settings.llm_rate_limit_max_retries,
                    str(e)[:300],
                )
                raise
    raise last_error  # type: ignore[misc]


def _extract_strategic_brief(result) -> StrategicBrief:
    """Extract StrategicBrief from CrewAI result. Handles different CrewAI versions."""
    if hasattr(result, "pydantic") and result.pydantic is not None:
        return result.pydantic
    if hasattr(result, "tasks_output") and result.tasks_output:
        last = result.tasks_output[-1]
        if isinstance(last, StrategicBrief):
            return last
        if hasattr(last, "output"):
            out = last.output
            if isinstance(out, StrategicBrief):
                return out
            if isinstance(out, str):
                return StrategicBrief.model_validate_json(out)
    if hasattr(result, "raw") and isinstance(result.raw, str):
        try:
            return StrategicBrief.model_validate_json(result.raw)
        except Exception:
            pass
    raise ValueError(
        "Could not extract StrategicBrief from crew result. "
        "Check CrewAI version and result structure."
    )


class StratAgentCrew:
    def __init__(self):
        self.research_agent = create_research_agent()
        self.critic_agent = create_critic_agent()
        self.synthesis_agent = create_synthesis_agent()

    def run(self, company: str, question: str) -> StrategicBrief:
        logger.info("Starting StratAgent analysis for %s", company)

        research_task = create_research_task(self.research_agent, company, question)
        critic_task = create_critic_task(self.critic_agent, company, question, research_task)
        synthesis_task = create_synthesis_task(
            self.synthesis_agent, company, question, research_task, critic_task
        )

        crew = Crew(
            agents=[self.research_agent, self.critic_agent, self.synthesis_agent],
            tasks=[research_task, critic_task, synthesis_task],
            process=Process.sequential,
            verbose=True,
            memory=True,
            embedder={
                "provider": "google-generativeai",
                "config": {
                    "model_name": "gemini-embedding-001",
                    "api_key": settings.gemini_api_key,
                },
            },
            max_rpm=30,
        )

        result = _run_crew_with_retry(crew, {"company": company, "question": question})
        brief = _extract_strategic_brief(result)

        logger.info("Analysis complete for %s", company)
        return brief
