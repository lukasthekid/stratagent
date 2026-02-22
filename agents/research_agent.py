from crewai import Agent, LLM

from agents.tools.retrieval_tool import RetrievalTool
from agents.tools.search_tool import WebSearchTool
from config import settings


def create_research_agent() -> Agent:
    llm = LLM(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=None,
        timeout=None,
    )

    return Agent(
        role="Senior Research Analyst",
        goal=(
            "Gather comprehensive, verified information about a company's strategic position, "
            "market context, and recent developments. Every claim must be backed by a source."
        ),
        backstory=(
            "You are a seasoned research analyst with 15 years of experience at top-tier "
            "consulting firms. You have a reputation for never presenting unverified information. "
            "You always cite your sources and flag when information is uncertain or outdated. "
            "You focus on finding information that is strategically relevant, not just factually interesting."
        ),
        tools=[RetrievalTool(), WebSearchTool()],
        llm=llm,
        verbose=True,
        max_iter=2,  # Prevents infinite tool loops
        memory=True,
    )
