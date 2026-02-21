"""API tests."""

from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient

from agents.schemas import (
    FinancialAnalysis,
    ResearchFindings,
    StrategicBrief,
    SWOTAnalysis,
)
from api.main import app

client = TestClient(app)


def test_health_check() -> None:
    """Test health endpoint returns ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyse_validates_request_body() -> None:
    """Test /analyse requires company and query."""
    response = client.post("/analyse", json={})
    assert response.status_code == 422  # Validation error


def test_analyse_accepts_valid_request() -> None:
    """Test /analyse accepts valid request (mocked crew)."""
    mock_brief = StrategicBrief(
        company="Acme Corp",
        executive_summary="Test summary",
        research_findings=ResearchFindings(
            company="Acme",
            key_facts=[],
            sources=[],
            market_context="",
            confidence_score=0.9,
        ),
        financial_analysis=FinancialAnalysis(
            revenue_trend="",
            profit_margins="",
            key_ratios={},
            growth_rates={},
            financial_risks=[],
            financial_summary="",
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
    with patch("api.main.StratAgentCrew") as MockCrew:
        MockCrew.return_value.run.return_value = mock_brief
        response = client.post(
            "/analyse",
            json={"company": "Acme Corp", "query": "What are the growth opportunities?"},
        )
    assert response.status_code == 200
    assert response.json()["company"] == "Acme Corp"


def test_ingest_upload_requires_files() -> None:
    """Test /ingest/upload requires at least one file."""
    response = client.post("/ingest/upload")  # No files sent
    assert response.status_code == 422  # Validation error


def test_ingest_upload_rejects_unsupported_type() -> None:
    """Test /ingest/upload rejects unsupported file types."""
    response = client.post(
        "/ingest/upload",
        files=[("files", ("test.txt", BytesIO(b"hello"), "text/plain"))],
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]


def test_ingest_upload_accepts_pdf() -> None:
    """Test /ingest/upload accepts PDF and returns chunk info (mocked)."""
    mock_docs = [type("Doc", (), {"page_content": "test", "metadata": {}})()]
    with (
        patch("api.main.load_documents", return_value=mock_docs),
        patch("api.main.upsert_documents", return_value=["id1", "id2"]),
    ):
        response = client.post(
            "/ingest/upload",
            files=[("files", ("report.pdf", BytesIO(b"%PDF-1.4 fake"), "application/pdf"))],
        )
    assert response.status_code == 200
    data = response.json()
    assert data["chunk_count"] == 2
    assert data["chunk_ids"] == ["id1", "id2"]
    assert len(data["files"]) == 1
    assert data["files"][0]["filename"] == "report.pdf"
    assert data["files"][0]["documents"] == 1
    assert data["files"][0]["chunks"] == 2


def test_ingest_url_validates_url() -> None:
    """Test /ingest/url requires valid URL."""
    response = client.post("/ingest/url", json={"url": "not-a-url"})
    assert response.status_code == 400


def test_ingest_url_accepts_valid_url() -> None:
    """Test /ingest/url accepts valid URL (mocked)."""
    mock_docs = [type("Doc", (), {"page_content": "page", "metadata": {}})()]
    with (
        patch("api.main.load_documents", return_value=mock_docs),
        patch("api.main.upsert_documents", return_value=["id1"]),
    ):
        response = client.post(
            "/ingest/url",
            json={"url": "https://example.com/page"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["chunk_count"] == 1
    assert data["chunk_ids"] == ["id1"]
