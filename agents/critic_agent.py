from crewai import Agent, LLM

from agents.schemas import CritiqueReport
from agents.tools.search_tool import WebSearchTool
from config import settings


def create_critic_agent() -> Agent:
    """Lightweight critic: short prompt, web search for counterevidence, concise bullet-point output."""
    llm = LLM(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.2,
        max_tokens=settings.max_tokens,
        timeout=None,
        max_retries=settings.llm_rate_limit_max_retries,
    )

    return Agent(
        role="Critical Reviewer",
        goal="Stress-test research: flag weak claims, gaps, assumptions. Brief bullet points.",
        backstory="Skeptical analyst. Direct and concise.",
        tools=[WebSearchTool()],
        verbose=True,
        max_iter=4,  # Allow web search for counterevidence + synthesis
        memory=True,
        llm=llm,
        output_pydantic=CritiqueReport,
    )