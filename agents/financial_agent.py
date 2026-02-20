from crewai import Agent, LLM
from agents.tools.retrieval_tool import RetrievalTool
from agents.tools.calculator_tool import FinancialCalculatorTool
from config import settings


def create_financial_agent() -> Agent:
    llm = LLM(
        model=settings.llm_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=None,
        timeout=None,
    )

    return Agent(
        role="Senior Financial Analyst",
        goal=(
            "Analyze the company's financial performance with precision. "
            "Always compute ratios using the calculator tool — never estimate. "
            "Identify trends, risks, and anomalies in the financial data."
        ),
        backstory=(
            "You are a CFA charterholder with deep expertise in financial statement analysis. "
            "You worked at Goldman Sachs for 10 years before moving into strategy consulting. "
            "You are obsessed with accuracy — you never quote a number without verifying it "
            "and you always show your calculation methodology. You flag data quality issues."
        ),
        tools=[RetrievalTool(), FinancialCalculatorTool()],
        llm=llm,
        verbose=True,
        max_iter=2,
        memory=True,
    )