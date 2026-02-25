from crewai import Task
from agents.schemas import ResearchFindings, StrategicBrief, CritiqueReport


def create_research_task(agent, company: str, question: str) -> Task:
    return Task(
        description=f"""Research {company} for: "{question}"
Cover: strategic position, recent news (12mo), top 3 competitors, key initiatives, market trends.
1. Use Document Retrieval for company facts. 2. Use Web Search for news/competitors. 3. Output raw JSON (not markdown).
Required: company, key_facts (4-6), sources, market_context (1-2 paragraphs), confidence_score (0.0-1.0).""",
        expected_output='Raw JSON: {{"company": "{company}", "key_facts": ["..."], "sources": ["..."], "market_context": "...", "confidence_score": 0.8}}',
        agent=agent,
        output_pydantic=ResearchFindings,
    )


def create_critic_task(agent, company: str, question: str, research_task) -> Task:
    return Task(
        description=f"""Review research on {company} for: "{question}". Return 1-2 bullet points each: well-supported claims, weak claims, gaps, counterarguments, key assumptions. Overall quality: Strong/Adequate/Weak.""",
        expected_output="Brief CritiqueReport: 1-2 items per list. overall_research_quality: Strong/Adequate/Weak.",
        agent=agent,
        output_pydantic=CritiqueReport,
        context=[research_task],
    )


def create_synthesis_task(
    agent, company: str, question: str, research_task, critic_task
) -> Task:
    return Task(
        description=f"""Synthesize research and critique for {company} into a strategic brief answering: "{question}". Include: 3-sentence executive summary, SWOT (3 points per quadrant), top 3 risks, 3 recommendations, caveats, confidence (High/Medium/Low). Ground in research; use critique for gaps.""",
        expected_output="StrategicBrief with all fields populated.",
        agent=agent,
        output_pydantic=StrategicBrief,
        context=[research_task, critic_task],
    )