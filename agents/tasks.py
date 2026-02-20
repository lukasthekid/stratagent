from crewai import Task
from agents.schemas import ResearchFindings, FinancialAnalysis, StrategicBrief


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
        expected_output=f"""
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


def create_financial_task(agent, company: str, research_task) -> Task:
    return Task(
        description=f"""
        Perform a rigorous financial analysis of {company} using the research findings provided
        and your own document retrieval.

        You must analyze:
        1. Revenue and profit trends over the last 3 years
        2. Key financial ratios: gross margin, net margin, debt-to-equity, current ratio
        3. YoY growth rates for revenue, EBITDA, and free cash flow
        4. Capital allocation strategy (R&D spend, capex, buybacks)
        5. Financial risks and red flags

        RULES:
        - Use the Financial Calculator Tool for every ratio and growth rate — never estimate
        - Retrieve raw financial data from documents before calculating
        - Flag any inconsistencies in the data
        - If data is unavailable, state this explicitly rather than guessing
        """,
        expected_output="""
        A structured financial analysis in JSON format covering:
        - revenue_trend: narrative description
        - profit_margins: current margins with context
        - key_ratios: dictionary of ratio names to values
        - growth_rates: dictionary of metric names to YoY growth rates
        - financial_risks: list of identified risks
        - financial_summary: 1-paragraph executive summary
        """,
        agent=agent,
        output_pydantic=FinancialAnalysis,
        context=[research_task],
    )


def create_synthesis_task(agent, company: str, question: str, research_task, financial_task) -> Task:
    return Task(
        description=f"""
        You have received research findings and financial analysis for {company}.
        Your job is to synthesize these into a definitive strategic brief that answers:
        "{question}"

        Your brief must include:
        1. A 3-sentence executive summary that directly answers the question
        2. A rigorous SWOT analysis (3-5 points per quadrant, no generic filler)
        3. The top 3-5 strategic risks, ranked by likelihood × impact
        4. 3-5 specific, actionable recommendations with rationale
        5. An overall confidence assessment (High/Medium/Low) with justification

        RULES:
        - Every point in the SWOT must be grounded in the research or financial findings
        - Recommendations must be specific — avoid vague statements like "invest in innovation"
        - The executive summary must be written for a CEO, not an analyst
        - Flag any areas where data quality was poor
        """,
        expected_output="A complete StrategicBrief object with all fields populated.",
        agent=agent,
        output_pydantic=StrategicBrief,
        context=[research_task, financial_task],  # Receives both prior outputs
    )