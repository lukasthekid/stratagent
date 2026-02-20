"""Pytest fixtures for stratagent tests."""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Use project fixtures dir to avoid tmp_path PermissionError on Windows
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def sample_pdf_path() -> Path:
    """Create a minimal PDF with extractable text for testing."""
    FIXTURES_DIR.mkdir(exist_ok=True)
    pdf_path = FIXTURES_DIR / "sample.pdf"
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
def sample_csv_path() -> Path:
    """Create a sample financial CSV for testing."""
    FIXTURES_DIR.mkdir(exist_ok=True)
    csv_path = FIXTURES_DIR / "financials.csv"
    csv_path.write_text(
        "date,revenue,net_income,total_assets,total_liabilities\n"
        "2024-01-01,1000000,150000,5000000,2000000\n"
        "2024-02-01,1200000,180000,5200000,2100000\n"
        "2024-03-01,1100000,165000,5100000,2050000\n",
        encoding="utf-8",
    )
    return csv_path


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]
