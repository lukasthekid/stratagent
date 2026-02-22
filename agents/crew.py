import logging

import mlflow
from crewai import Crew, Process

from agents.critic_agent import create_critic_agent
from agents.research_agent import create_research_agent
from agents.schemas import StrategicBrief
from agents.synthesis_agent import create_synthesis_agent
from agents.tasks import create_research_task, create_synthesis_task, create_critic_task
from config import settings

logger = logging.getLogger(__name__)


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

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        exp = mlflow.get_experiment_by_name(settings.mlflow_experiment_name)
        if exp is None:
            mlflow.create_experiment(settings.mlflow_experiment_name)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name=f"{company}"):
            mlflow.log_params({
                "company": company,
                "model": settings.llm_model,
                "retrieval_k": str(settings.retrieval_top_k),
                "rerank_k": str(settings.rerank_top_k),
                "chunk_size": str(settings.chunk_size),
            })

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

            result = crew.kickoff(inputs={"company": company, "question": question})
            brief = _extract_strategic_brief(result)

            conf_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
            cl = brief.confidence_level.lower() if brief.confidence_level else "medium"
            mlflow.log_metric("confidence_level", conf_map.get(cl, 0.5))

        logger.info("Analysis complete for %s", company)
        return brief
