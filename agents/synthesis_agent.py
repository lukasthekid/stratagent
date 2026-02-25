from crewai import Agent, LLM

from agents.schemas import StrategicBrief
from config import settings


def create_synthesis_agent() -> Agent:
    llm = LLM(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=settings.max_tokens,
        timeout=None,
        max_retries=settings.llm_rate_limit_max_retries,
    )

    return Agent(
        role="Principal Strategy Consultant",
        goal="Synthesize research and critique into strategic brief. SWOT, risks, recommendations.",
        backstory="Strategy consultant. Direct, actionable.",
        tools=[],  # Synthesis agent reasons only — no tools needed
        llm=llm,
        verbose=True,
        max_iter=2,
        memory=True,
        output_pydantic=StrategicBrief,
    )