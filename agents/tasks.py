from crewai import Task
from agents.schemas import ResearchFindings, StrategicBrief, CritiqueReport


def create_research_task(agent, company: str, question: str) -> Task:
    return Task(
        description=f"""
        Conduct comprehensive research on {company} to answer this strategic question:
        "{question}"

        Your research must cover:
        1. Company overview and current strategic position
        2. Recent news and developments (last 12 months)
        3. Competitive landscape — who are the top 3 competitors and how does {company} compare?
        4. Key strategic initiatives the company has announced
        5. Market trends affecting the company's industry

        RULES:
        - Use the Document Retrieval Tool for every factual claim about the company
        - Use Web Search for recent news and competitor information
        - Cite every source you use
        - If you cannot find evidence for a claim, say so explicitly
        - Do not hallucinate statistics
        """,
        expected_output="""
        A structured research report in JSON format matching this schema:
        - company: company name
        - key_facts: list of 8-12 verified facts with sources
        - sources: list of all documents and URLs cited
        - market_context: 2-3 paragraph summary of industry dynamics
        - confidence_score: float between 0 and 1
        """,
        agent=agent,
        output_pydantic=ResearchFindings,
    )


def create_critic_task(agent, company: str, question: str, research_task) -> Task:
    return Task(
        description=f"""
        You have received research findings about {company} in response to this question:
        "{question}"

        Your job is to critically review these findings across four dimensions:

        1. EVIDENCE QUALITY — Which claims are well-supported? Which are asserted without 
           strong evidence? Flag any statistics or facts that seem unverified.

        2. GAPS — What important aspects of the question were NOT addressed by the research?
           What would a skeptical executive ask that the research can't currently answer?

        3. COUNTERARGUMENTS — What is the strongest case AGAINST the research's implied 
           conclusions? Search the web if needed to find opposing viewpoints or contradicting data.

        4. RISKS & ASSUMPTIONS — What assumptions is the research implicitly making? 
           What would have to be true for the conclusions to be wrong?

        Be direct and specific. Do not soften your critique to be polite.
        """,
        expected_output="""
        A structured critique containing:
        - well_supported_claims: list of findings that are strongly evidenced
        - weak_or_unsupported_claims: list of findings that need stronger backing
        - gaps: list of important topics the research missed
        - counterarguments: list of opposing viewpoints or contradicting evidence
        - key_assumptions: list of implicit assumptions in the research
        - overall_research_quality: "Strong" / "Adequate" / "Weak" with 1-sentence justification
        """,
        agent=agent,
        output_pydantic=CritiqueReport,
        context=[research_task],
    )


def create_synthesis_task(
    agent, company: str, question: str, research_task, critic_task
) -> Task:
    return Task(
        description=f"""
        You have received research findings and a critical review for {company}.
        Your job is to synthesize both into a definitive strategic brief that answers:
        "{question}"

        The critical review identifies gaps, weak evidence, counterarguments, and assumptions.
        Use it to strengthen your brief: address gaps in your recommendations, incorporate
        counterarguments into strategic risks, and adjust confidence based on evidence quality.

        Your brief must include:
        1. A 3-sentence executive summary that directly answers the question
        2. A rigorous SWOT analysis (3-5 points per quadrant, no generic filler)
        3. The top 3-5 strategic risks, ranked by likelihood × impact
        4. 3-5 specific, actionable recommendations with rationale
        5. Caveats: limitations, data gaps, assumptions, or disclaimers the reader should know
        6. An overall confidence assessment (High/Medium/Low) with justification

        RULES:
        - Every point in the SWOT must be grounded in the research findings
        - Incorporate the critique: address gaps, factor in counterarguments, reflect evidence quality
        - Recommendations must be specific — avoid vague statements like "invest in innovation"
        - The executive summary must be written for a CEO, not an analyst
        - Flag any areas where data quality was poor
        """,
        expected_output="A complete StrategicBrief object with all fields populated.",
        agent=agent,
        output_pydantic=StrategicBrief,
        context=[research_task, critic_task],
    )