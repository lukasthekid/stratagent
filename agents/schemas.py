from pydantic import BaseModel, Field
from typing import List, Optional

class ResearchFindings(BaseModel):
    company: str
    key_facts: List[str] = Field(description="List of verified facts from retrieved documents")
    sources: List[str] = Field(description="Source documents cited")
    market_context: str = Field(description="Industry and market context summary")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in findings based on source quality")

class FinancialAnalysis(BaseModel):
    revenue_trend: str
    profit_margins: str
    key_ratios: dict = Field(description="Important financial ratios with values")
    growth_rates: dict = Field(description="YoY growth rates for key metrics")
    financial_risks: List[str]
    financial_summary: str

class SWOTAnalysis(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]

class StrategicBrief(BaseModel):
    company: str
    executive_summary: str
    research_findings: ResearchFindings
    financial_analysis: FinancialAnalysis
    swot: SWOTAnalysis
    strategic_risks: List[str] = Field(description="Top 3-5 prioritized strategic risks")
    recommendations: List[str] = Field(description="Actionable strategic recommendations")
    confidence_level: str = Field(description="Overall confidence: High / Medium / Low")