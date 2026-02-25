import json
from typing import Any, List

from pydantic import BaseModel, Field, field_validator


def _extract_json_object(s: str) -> str:
    """Extract valid JSON from LLM output, stripping markdown and trailing text."""
    s = s.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines)
    try:
        json.loads(s)
        return s
    except json.JSONDecodeError:
        pass
    for i in range(len(s) - 1, -1, -1):
        if s[i] == "}":
            try:
                json.loads(s[: i + 1])
                return s[: i + 1]
            except json.JSONDecodeError:
                continue
    return s


class ResearchFindings(BaseModel):
    company: str
    key_facts: List[str] = Field(description="Verified facts")
    sources: List[str] = Field(description="Sources cited")
    market_context: str = Field(description="Market context summary")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence 0-1")

    @classmethod
    def model_validate_json(cls, s: str | bytes, **kwargs):
        raw = s.decode() if isinstance(s, bytes) else s
        return super().model_validate_json(_extract_json_object(raw), **kwargs)

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

    @classmethod
    def model_validate_json(cls, s: str | bytes, **kwargs):
        """Parse JSON from LLM output, tolerating trailing characters and markdown."""
        raw = s.decode() if isinstance(s, bytes) else s
        cleaned = _extract_json_object(raw)
        return super().model_validate_json(cleaned, **kwargs)

class StrategicBrief(BaseModel):
    company: str
    executive_summary: str
    research_findings: ResearchFindings
    swot: SWOTAnalysis
    strategic_risks: List[str] = Field(description="Top strategic risks")
    recommendations: List[str] = Field(description="Actionable recommendations")
    caveats: List[str] = Field(description="Limitations, gaps, assumptions")
    confidence_level: str = Field(description="High/Medium/Low")

    @classmethod
    def model_validate_json(cls, s: str | bytes, **kwargs):
        raw = s.decode() if isinstance(s, bytes) else s
        return super().model_validate_json(_extract_json_object(raw), **kwargs)

    @field_validator("strategic_risks", mode="before")
    @classmethod
    def coerce_strategic_risks(cls, v: Any) -> List[str]:
        """Coerce LLM output: accept dicts with 'risk' key and extract the string."""
        if not isinstance(v, list):
            return v
        result: List[str] = []
        for item in v:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict) and "risk" in item:
                result.append(str(item["risk"]))
            else:
                result.append(str(item))
        return result