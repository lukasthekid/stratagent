import logging

from crewai import Crew, Process

from agents.financial_agent import create_financial_agent
from agents.research_agent import create_research_agent
from agents.schemas import StrategicBrief
from agents.synthesis_agent import create_synthesis_agent
from agents.tasks import create_financial_task, create_research_task, create_synthesis_task

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
        self.financial_agent = create_financial_agent()
        self.synthesis_agent = create_synthesis_agent()

    def run(self, company: str, question: str) -> StrategicBrief:
        logger.info("Starting StratAgent analysis for %s", company)

        research_task = create_research_task(self.research_agent, company, question)
        financial_task = create_financial_task(self.financial_agent, company, research_task)
        synthesis_task = create_synthesis_task(
            self.synthesis_agent, company, question, research_task, financial_task
        )

        crew = Crew(
            agents=[self.research_agent, self.financial_agent, self.synthesis_agent],
            tasks=[research_task, financial_task, synthesis_task],
            process=Process.sequential,
            verbose=True,
            memory=True,
            max_rpm=30,
        )

        result = crew.kickoff(inputs={"company": company, "question": question})

        logger.info("Analysis complete for %s", company)
        return _extract_strategic_brief(result)