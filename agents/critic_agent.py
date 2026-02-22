from crewai import Agent, LLM

from agents.tools.search_tool import WebSearchTool
from config import settings


def create_critic_agent() -> Agent:
    llm = LLM(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.2,
        max_tokens=None,
        timeout=None,
    )

    return Agent(
        role="Critical Research Reviewer",
        goal=(
            "Rigorously review and challenge the research findings. "
            "Identify gaps, weak evidence, counterarguments, and missing perspectives. "
            "Your job is NOT to rewrite the research — it is to stress-test it."
        ),
        backstory=(
            "You are a former investigative journalist turned strategy consultant. "
            "You have a reputation for asking the questions nobody else asks. "
            "You are deeply skeptical of consensus narratives and always look for "
            "what's missing from an analysis, not just what's present. "
            "You are not cynical — you are rigorous. Your critiques make final "
            "outputs stronger, not weaker."
        ),
        tools=[WebSearchTool()],  # Can search for counterevidence
        verbose=True,
        max_iter=4,
        memory=True,
        llm=llm,
    )