from crewai import Agent, LLM
from langchain_groq import ChatGroq

from config import settings


def create_synthesis_agent() -> Agent:
    llm = LLM(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=None,
        timeout=None,
    )

    return Agent(
        role="Principal Strategy Consultant",
        goal=(
            "Synthesize research findings and critical review into a crisp, actionable strategic brief. "
            "Produce a structured SWOT analysis and clear recommendations that a C-suite executive "
            "can act on immediately. Prioritize insight over information."
        ),
        backstory=(
            "You are a Principal Consultant at a top-3 strategy firm with 20 years of experience "
            "delivering board-level recommendations to Fortune 500 companies. "
            "You have a gift for cutting through complexity to identify the two or three things "
            "that actually matter. Your briefs are known for being direct, structured, and "
            "immediately actionable. You never pad your output with filler."
        ),
        tools=[],  # Synthesis agent reasons only — no tools needed
        llm=llm,
        verbose=True,
        max_iter=2,
        memory=True,
    )