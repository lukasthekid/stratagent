from pydantic import BaseModel, Field
from typing import List, Optional

class ResearchFindings(BaseModel):
    company: str
    key_facts: List[str] = Field(description="List of verified facts from retrieved documents")
    sources: List[str] = Field(description="Source documents cited")
    market_context: str = Field(description="Industry and market context summary")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in findings based on source quality")

class SWOTAnalysis(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    opportunities: List[str]
    threats: List[str]

class CritiqueReport(BaseModel):
    well_supported_claims: List[str]
    weak_or_unsupported_claims: List[str]
    gaps: List[str]
    counterarguments: List[str]
    key_assumptions: List[str]
    overall_research_quality: str  # "Strong" / "Adequate" / "Weak"

class StrategicBrief(BaseModel):
    company: str
    executive_summary: str
    research_findings: ResearchFindings
    swot: SWOTAnalysis
    strategic_risks: List[str] = Field(description="Top 3-5 prioritized strategic risks")
    recommendations: List[str] = Field(description="Actionable strategic recommendations")
    caveats: List[str] = Field(
        description="Limitations, data gaps, assumptions, or disclaimers the reader should know"
    )
    confidence_level: str = Field(description="Overall confidence: High / Medium / Low")