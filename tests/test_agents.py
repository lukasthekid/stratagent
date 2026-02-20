"""Tests for StratAgent crew, agents, tasks, schemas, and tools."""

from unittest.mock import MagicMock, patch

import pytest
from crewai import Agent, Task
from langchain_core.documents import Document
from pydantic import ValidationError

from agents.crew import StratAgentCrew, _extract_strategic_brief
from agents.financial_agent import create_financial_agent
from agents.research_agent import create_research_agent
from agents.schemas import (
    FinancialAnalysis,
    ResearchFindings,
    StrategicBrief,
    SWOTAnalysis,
)
from agents.synthesis_agent import create_synthesis_agent
from agents.tasks import (
    create_financial_task,
    create_research_task,
    create_synthesis_task,
)
from agents.tools.calculator_tool import FinancialCalculatorTool
from agents.tools.retrieval_tool import RetrievalTool
from agents.tools.search_tool import WebSearchTool


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TestResearchFindings:
    """Tests for ResearchFindings schema."""

    def test_valid_research_findings(self) -> None:
        findings = ResearchFindings(
            company="Acme Corp",
            key_facts=["Fact 1", "Fact 2"],
            sources=["doc1.pdf", "doc2.pdf"],
            market_context="Industry is growing.",
            confidence_score=0.85,
        )
        assert findings.company == "Acme Corp"
        assert len(findings.key_facts) == 2
        assert findings.confidence_score == 0.85

    def test_confidence_score_bounds(self) -> None:
        ResearchFindings(
            company="X",
            key_facts=[],
            sources=[],
            market_context="",
            confidence_score=0.0,
        )
        ResearchFindings(
            company="X",
            key_facts=[],
            sources=[],
            market_context="",
            confidence_score=1.0,
        )

    def test_confidence_score_below_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            ResearchFindings(
                company="X",
                key_facts=[],
                sources=[],
                market_context="",
                confidence_score=-0.1,
            )

    def test_confidence_score_above_one_raises(self) -> None:
        with pytest.raises(ValidationError):
            ResearchFindings(
                company="X",
                key_facts=[],
                sources=[],
                market_context="",
                confidence_score=1.1,
            )


class TestFinancialAnalysis:
    """Tests for FinancialAnalysis schema."""

    def test_valid_financial_analysis(self) -> None:
        analysis = FinancialAnalysis(
            revenue_trend="Growing",
            profit_margins="Stable at 15%",
            key_ratios={"gross_margin": "45%", "net_margin": "12%"},
            growth_rates={"revenue": "8%", "ebitda": "10%"},
            financial_risks=["Debt levels rising"],
            financial_summary="Solid performance.",
        )
        assert analysis.revenue_trend == "Growing"
        assert "gross_margin" in analysis.key_ratios
        assert len(analysis.financial_risks) == 1


class TestSWOTAnalysis:
    """Tests for SWOTAnalysis schema."""

    def test_valid_swot(self) -> None:
        swot = SWOTAnalysis(
            strengths=["Strong brand"],
            weaknesses=["High costs"],
            opportunities=["Market expansion"],
            threats=["Competition"],
        )
        assert len(swot.strengths) == 1
        assert len(swot.weaknesses) == 1


class TestStrategicBrief:
    """Tests for StrategicBrief schema."""

    def test_valid_strategic_brief(self) -> None:
        brief = StrategicBrief(
            company="Acme",
            executive_summary="Summary here.",
            research_findings=ResearchFindings(
                company="Acme",
                key_facts=[],
                sources=[],
                market_context="",
                confidence_score=0.8,
            ),
            financial_analysis=FinancialAnalysis(
                revenue_trend="Up",
                profit_margins="15%",
                key_ratios={},
                growth_rates={},
                financial_risks=[],
                financial_summary="Good.",
            ),
            swot=SWOTAnalysis(
                strengths=[],
                weaknesses=[],
                opportunities=[],
                threats=[],
            ),
            strategic_risks=["Risk 1"],
            recommendations=["Rec 1"],
            confidence_level="High",
        )
        assert brief.company == "Acme"
        assert brief.confidence_level == "High"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class TestRetrievalTool:
    """Tests for RetrievalTool."""

    @patch("agents.tools.retrieval_tool.retrieve_with_rerank")
    def test_returns_formatted_documents(
        self, mock_retrieve: MagicMock, sample_documents: list[Document]
    ) -> None:
        mock_retrieve.invoke.return_value = sample_documents

        tool = RetrievalTool()
        result = tool._run(query="pet animals")

        mock_retrieve.invoke.assert_called_once_with(
            {"query": "pet animals", "retrieval_k": 50, "rerank_k": 5}
        )
        assert "Dogs are great companions" in result
        assert "Source: mammal-pets-doc" in result

    @patch("agents.tools.retrieval_tool.retrieve_with_rerank")
    def test_empty_results_returns_message(self, mock_retrieve: MagicMock) -> None:
        mock_retrieve.invoke.return_value = []

        tool = RetrievalTool()
        result = tool._run(query="obscure query")

        assert "No relevant documents found" in result

    @patch("agents.tools.retrieval_tool.retrieve_with_rerank")
    def test_retrieval_error_returns_error_message(self, mock_retrieve: MagicMock) -> None:
        mock_retrieve.invoke.side_effect = Exception("Connection failed")

        tool = RetrievalTool()
        result = tool._run(query="test")

        assert "Retrieval error" in result
        assert "Connection failed" in result


class TestFinancialCalculatorTool:
    """Tests for FinancialCalculatorTool."""

    def test_growth_rate(self) -> None:
        tool = FinancialCalculatorTool()
        result = tool._run(
            data={"current": 120, "previous": 100},
            calculations=["growth_rate"],
        )
        assert "20.00%" in result

    def test_gross_margin(self) -> None:
        tool = FinancialCalculatorTool()
        result = tool._run(
            data={"revenue": 1000, "cogs": 400},
            calculations=["gross_margin"],
        )
        assert "60.00%" in result

    def test_net_margin(self) -> None:
        tool = FinancialCalculatorTool()
        result = tool._run(
            data={"revenue": 1000, "net_income": 150},
            calculations=["net_margin"],
        )
        assert "15.00%" in result

    def test_current_ratio(self) -> None:
        tool = FinancialCalculatorTool()
        result = tool._run(
            data={"current_assets": 500, "current_liabilities": 200},
            calculations=["current_ratio"],
        )
        assert "2.50" in result

    def test_debt_to_equity(self) -> None:
        tool = FinancialCalculatorTool()
        result = tool._run(
            data={"total_debt": 300, "equity": 600},
            calculations=["debt_to_equity"],
        )
        assert "0.50" in result

    def test_multiple_calculations(self) -> None:
        tool = FinancialCalculatorTool()
        result = tool._run(
            data={
                "revenue": 1000,
                "cogs": 400,
                "net_income": 100,
                "current": 110,
                "previous": 100,
            },
            calculations=["gross_margin", "net_margin", "growth_rate"],
        )
        assert "60.00%" in result
        assert "10.00%" in result
        assert "10.00%" in result

    def test_division_by_zero_returns_na(self) -> None:
        tool = FinancialCalculatorTool()
        result = tool._run(
            data={"current": 100, "previous": 0},
            calculations=["growth_rate"],
        )
        assert "N/A" in result

    def test_missing_data_returns_errors(self) -> None:
        tool = FinancialCalculatorTool()
        result = tool._run(
            data={"revenue": 1000},
            calculations=["gross_margin", "net_margin"],
        )
        assert "Missing inputs" in result
        assert "gross_margin" in result or "cogs" in result

    def test_invalid_data_type_returns_error(self) -> None:
        tool = FinancialCalculatorTool()
        result = tool._run(data="not a dict", calculations=["growth_rate"])
        assert "Error" in result
        assert "dictionary" in result

    def test_invalid_calculations_type_returns_error(self) -> None:
        tool = FinancialCalculatorTool()
        result = tool._run(data={}, calculations="not a list")
        assert "Error" in result
        assert "list" in result


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    @patch("agents.tools.search_tool.TavilyClient")
    def test_returns_formatted_results(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Test Article",
                    "url": "https://example.com",
                    "content": "Article content here.",
                }
            ]
        }
        mock_client_cls.return_value = mock_client

        tool = WebSearchTool()
        result = tool._run(query="Acme Corp news")

        mock_client.search.assert_called_once_with(
            query="Acme Corp news", max_results=5, search_depth="advanced"
        )
        assert "Test Article" in result
        assert "https://example.com" in result
        assert "Article content here" in result

    @patch("agents.tools.search_tool.TavilyClient")
    def test_empty_results(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        mock_client_cls.return_value = mock_client

        tool = WebSearchTool()
        result = tool._run(query="obscure")

        assert "No results found" in result

    @patch("agents.tools.search_tool.TavilyClient")
    def test_search_error_returns_message(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("API key invalid")
        mock_client_cls.return_value = mock_client

        tool = WebSearchTool()
        result = tool._run(query="test")

        assert "Web search error" in result
        assert "API key invalid" in result


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


class TestAgentCreation:
    """Tests for agent factory functions."""

    @patch("agents.research_agent.settings")
    def test_create_research_agent(self, mock_settings: MagicMock) -> None:
        mock_settings.llm_model = "groq/llama"
        mock_settings.groq_api_key = "test-key"

        agent = create_research_agent()

        assert isinstance(agent, Agent)
        assert "Research" in agent.role
        assert len(agent.tools) == 2
        tool_names = [t.name for t in agent.tools]
        assert "Document Retrieval Tool" in tool_names
        assert "Web Search Tool" in tool_names

    @patch("agents.financial_agent.settings")
    def test_create_financial_agent(self, mock_settings: MagicMock) -> None:
        mock_settings.llm_model = "groq/llama"
        mock_settings.groq_api_key = "test-key"

        agent = create_financial_agent()

        assert isinstance(agent, Agent)
        assert "Financial" in agent.role
        assert len(agent.tools) == 2
        tool_names = [t.name for t in agent.tools]
        assert "Document Retrieval Tool" in tool_names
        assert "Financial Calculator Tool" in tool_names

    @patch("agents.synthesis_agent.settings")
    def test_create_synthesis_agent(self, mock_settings: MagicMock) -> None:
        mock_settings.llm_model = "groq/llama"
        mock_settings.groq_api_key = "test-key"

        agent = create_synthesis_agent()

        assert isinstance(agent, Agent)
        assert "Strategy" in agent.role or "Consultant" in agent.role
        assert len(agent.tools) == 0


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class TestTaskCreation:
    """Tests for task factory functions."""

    @patch("agents.research_agent.settings")
    def test_create_research_task(self, mock_settings: MagicMock) -> None:
        mock_settings.llm_model = "groq/llama"
        mock_settings.groq_api_key = "test-key"
        agent = create_research_agent()

        task = create_research_task(agent, "Acme Corp", "What are the growth prospects?")

        assert isinstance(task, Task)
        assert "Acme Corp" in task.description
        assert "growth prospects" in task.description
        assert task.agent == agent
        assert task.output_pydantic == ResearchFindings

    @patch("agents.financial_agent.settings")
    @patch("agents.research_agent.settings")
    def test_create_financial_task(
        self,
        mock_research_settings: MagicMock,
        mock_financial_settings: MagicMock,
    ) -> None:
        mock_research_settings.llm_model = mock_financial_settings.llm_model = "groq/llama"
        mock_research_settings.groq_api_key = mock_financial_settings.groq_api_key = "key"
        research_agent = create_research_agent()
        financial_agent = create_financial_agent()
        research_task = create_research_task(research_agent, "Acme", "Q?")

        task = create_financial_task(financial_agent, "Acme Corp", research_task)

        assert isinstance(task, Task)
        assert "Acme Corp" in task.description
        assert task.context == [research_task]
        assert task.output_pydantic == FinancialAnalysis

    @patch("agents.synthesis_agent.settings")
    @patch("agents.financial_agent.settings")
    @patch("agents.research_agent.settings")
    def test_create_synthesis_task(
        self,
        mock_research_settings: MagicMock,
        mock_financial_settings: MagicMock,
        mock_synthesis_settings: MagicMock,
    ) -> None:
        for m in (mock_research_settings, mock_financial_settings, mock_synthesis_settings):
            m.llm_model = "groq/llama"
            m.groq_api_key = "key"
        research_agent = create_research_agent()
        financial_agent = create_financial_agent()
        synthesis_agent = create_synthesis_agent()
        research_task = create_research_task(research_agent, "Acme", "Q?")
        financial_task = create_financial_task(financial_agent, "Acme", research_task)

        task = create_synthesis_task(
            synthesis_agent, "Acme Corp", "Strategic outlook?", research_task, financial_task
        )

        assert isinstance(task, Task)
        assert "Acme Corp" in task.description
        assert "Strategic outlook" in task.description
        assert task.context == [research_task, financial_task]
        assert task.output_pydantic == StrategicBrief


# ---------------------------------------------------------------------------
# Crew and _extract_strategic_brief
# ---------------------------------------------------------------------------


def _make_sample_brief() -> StrategicBrief:
    return StrategicBrief(
        company="Acme",
        executive_summary="Summary.",
        research_findings=ResearchFindings(
            company="Acme",
            key_facts=[],
            sources=[],
            market_context="",
            confidence_score=0.8,
        ),
        financial_analysis=FinancialAnalysis(
            revenue_trend="Up",
            profit_margins="15%",
            key_ratios={},
            growth_rates={},
            financial_risks=[],
            financial_summary="Good.",
        ),
        swot=SWOTAnalysis(
            strengths=[],
            weaknesses=[],
            opportunities=[],
            threats=[],
        ),
        strategic_risks=[],
        recommendations=[],
        confidence_level="High",
    )


class TestExtractStrategicBrief:
    """Tests for _extract_strategic_brief helper."""

    def test_extract_from_pydantic_attr(self) -> None:
        brief = _make_sample_brief()
        result = MagicMock()
        result.pydantic = brief

        extracted = _extract_strategic_brief(result)
        assert extracted == brief
        assert extracted.company == "Acme"

    def test_extract_from_tasks_output_last_is_brief(self) -> None:
        brief = _make_sample_brief()
        result = MagicMock()
        result.pydantic = None
        result.tasks_output = [MagicMock(), brief]

        extracted = _extract_strategic_brief(result)
        assert extracted == brief

    def test_extract_from_tasks_output_last_has_output_brief(self) -> None:
        brief = _make_sample_brief()
        last_task = MagicMock()
        last_task.output = brief
        result = MagicMock()
        result.pydantic = None
        result.tasks_output = [MagicMock(), last_task]

        extracted = _extract_strategic_brief(result)
        assert extracted == brief

    def test_extract_from_tasks_output_last_has_output_json_str(self) -> None:
        brief = _make_sample_brief()
        result = MagicMock()
        result.pydantic = None
        last_task = MagicMock()
        last_task.output = brief.model_dump_json()
        result.tasks_output = [MagicMock(), last_task]

        extracted = _extract_strategic_brief(result)
        assert extracted.company == brief.company

    def test_extract_from_raw_json(self) -> None:
        brief = _make_sample_brief()
        result = MagicMock()
        result.pydantic = None
        result.tasks_output = []
        result.raw = brief.model_dump_json()

        extracted = _extract_strategic_brief(result)
        assert extracted.company == brief.company

    def test_extract_fails_raises_value_error(self) -> None:
        result = MagicMock()
        result.pydantic = None
        result.tasks_output = []
        result.raw = None

        with pytest.raises(ValueError, match="Could not extract StrategicBrief"):
            _extract_strategic_brief(result)


class TestStratAgentCrew:
    """Tests for StratAgentCrew."""

    @patch("agents.crew.create_synthesis_task")
    @patch("agents.crew.create_financial_task")
    @patch("agents.crew.create_research_task")
    @patch("agents.crew.Crew")
    def test_run_returns_strategic_brief(
        self,
        mock_crew_cls: MagicMock,
        mock_create_research_task: MagicMock,
        mock_create_financial_task: MagicMock,
        mock_create_synthesis_task: MagicMock,
    ) -> None:
        mock_research_task = MagicMock()
        mock_financial_task = MagicMock()
        mock_synthesis_task = MagicMock()
        mock_create_research_task.return_value = mock_research_task
        mock_create_financial_task.return_value = mock_financial_task
        mock_create_synthesis_task.return_value = mock_synthesis_task

        mock_crew_instance = MagicMock()
        brief = _make_sample_brief()
        mock_result = MagicMock()
        mock_result.pydantic = brief
        mock_crew_instance.kickoff.return_value = mock_result
        mock_crew_cls.return_value = mock_crew_instance

        crew = StratAgentCrew()
        result = crew.run(company="Acme Corp", question="Growth outlook?")

        assert result == brief
        assert result.company == brief.company
        mock_create_research_task.assert_called_once()
        mock_create_financial_task.assert_called_once()
        mock_create_synthesis_task.assert_called_once()
        mock_crew_instance.kickoff.assert_called_once_with(
            inputs={"company": "Acme Corp", "question": "Growth outlook?"}
        )
