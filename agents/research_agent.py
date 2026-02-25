from crewai import Agent, LLM

from agents.tools.retrieval_tool import RetrievalTool
from agents.tools.search_tool import WebSearchTool
from config import settings


def create_research_agent() -> Agent:
    llm = LLM(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=settings.max_tokens,
        timeout=None,
        max_retries=settings.llm_rate_limit_max_retries,
    )

    return Agent(
        role="Senior Research Analyst",
        goal="Gather verified strategic info. Cite sources.",
        backstory="Research analyst. Cite sources, flag uncertainty.",
        tools=[RetrievalTool(), WebSearchTool()],
        llm=llm,
        verbose=True,
        max_iter=6,
        memory=True,
    )
