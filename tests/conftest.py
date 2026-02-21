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
