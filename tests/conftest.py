"""Pytest fixtures for stratagent tests."""

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """Create a minimal PDF with extractable text for testing."""
    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, "Annual Report 2024")
    c.drawString(100, 730, "Acme Corporation")
    c.drawString(100, 700, "Revenue: $1.2B | Net Income: $180M")
    c.showPage()
    c.drawString(100, 750, "Page 2 - Market Analysis")
    c.drawString(100, 700, "Strategic outlook remains positive.")
    c.save()
    return pdf_path


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    """Create a sample financial CSV for testing."""
    csv_path = tmp_path / "financials.csv"
    csv_path.write_text(
        "date,revenue,net_income,total_assets,total_liabilities\n"
        "2024-01-01,1000000,150000,5000000,2000000\n"
        "2024-02-01,1200000,180000,5200000,2100000\n"
        "2024-03-01,1100000,165000,5100000,2050000\n",
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture
def sample_csv_with_nulls(tmp_path: Path) -> Path:
    """Create CSV with empty/NaN values for schema validation tests."""
    csv_path = tmp_path / "partial.csv"
    csv_path.write_text(
        "date,revenue,net_income\n"
        "2024-01-01,1000,\n"
        "2024-02-01,,200\n"
        "2024-03-01,1500,300\n",
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture
def invalid_csv_schema() -> type:
    """Pydantic schema that will fail validation (requires non-null revenue)."""

    from pydantic import BaseModel

    class StrictFinancialRow(BaseModel):
        date: str
        revenue: float  # No Optional - must be present
        net_income: float

    return StrictFinancialRow
