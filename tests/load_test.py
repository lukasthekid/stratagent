"""Tests for document loading."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion import load_documents
from langchain_core.documents import Document

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestLoadPDF:
    """PDF loading tests."""

    def test_load_pdf_returns_documents(self, sample_pdf_path):
        docs = load_documents(sample_pdf_path)
        assert len(docs) == 2  # 2 pages
        assert "Annual Report 2024" in docs[0].page_content
        assert "Acme Corporation" in docs[0].page_content
        assert "Page 2 - Market Analysis" in docs[1].page_content
        assert "Strategic outlook" in docs[1].page_content

    def test_load_pdf_with_path_object(self, sample_pdf_path):
        docs = load_documents(Path(sample_pdf_path))
        assert len(docs) >= 1
        assert "Annual Report 2024" in docs[0].page_content

    def test_load_pdf_uppercase_extension(self, sample_pdf_path):
        pdf_upper = FIXTURES_DIR / "sample.PDF"
        pdf_upper.write_bytes(sample_pdf_path.read_bytes())
        docs = load_documents(pdf_upper)
        assert len(docs) >= 1
        assert "Annual Report 2024" in docs[0].page_content


class TestLoadCSV:
    """CSV loading tests."""

    def test_load_csv_returns_documents(self, sample_csv_path):
        docs = load_documents(sample_csv_path)
        assert len(docs) == 3  # 3 data rows
        content = " ".join(d.page_content for d in docs)
        assert "2024-01-01" in content
        assert "revenue" in content.lower() or "1000000" in content

    def test_load_csv_with_path_object(self, sample_csv_path):
        docs = load_documents(Path(sample_csv_path))
        assert len(docs) >= 1

    def test_load_csv_custom_encoding(self, sample_csv_path):
        docs = load_documents(sample_csv_path, csv_encoding="utf-8")
        assert len(docs) >= 1


class TestLoadWeb:
    """Web URL loading tests (mocked to avoid network)."""

    @patch("ingestion.load.WebBaseLoader")
    def test_load_webpage_returns_documents(self, mock_loader_class):
        mock_loader = MagicMock()
        mock_loader.load.return_value = [
            Document(page_content="Test Financial Report 2024", metadata={"source": "https://example.com"}),
        ]
        mock_loader_class.return_value = mock_loader

        docs = load_documents("https://example.com/report")

        mock_loader_class.assert_called_once_with(web_path="https://example.com/report")
        mock_loader.load.assert_called_once()
        assert len(docs) == 1
        assert "Financial Report" in docs[0].page_content

    @patch("ingestion.load.WebBaseLoader")
    def test_load_http_url(self, mock_loader_class):
        mock_loader = MagicMock()
        mock_loader.load.return_value = [Document(page_content="HTTP content", metadata={})]
        mock_loader_class.return_value = mock_loader

        docs = load_documents("http://example.com")
        assert len(docs) == 1
        assert docs[0].page_content == "HTTP content"

    @patch("ingestion.load.WebBaseLoader")
    def test_load_sec_gov_returns_documents(self, mock_loader_class):
        """SEC.gov URLs load via WebBaseLoader with web_path."""
        mock_loader = MagicMock()
        mock_loader.load.return_value = [Document(page_content="SEC filing content", metadata={})]
        mock_loader_class.return_value = mock_loader

        docs = load_documents("https://www.sec.gov/Archives/edgar/data/1318605/000162828026003952/tsla-20251231.htm")

        mock_loader_class.assert_called_once_with(web_path="https://www.sec.gov/Archives/edgar/data/1318605/000162828026003952/tsla-20251231.htm")
        assert len(docs) == 1
        assert "SEC filing" in docs[0].page_content


class TestLoadErrors:
    """Error handling tests."""

    def test_unsupported_extension_raises(self):
        FIXTURES_DIR.mkdir(exist_ok=True)
        bad_file = FIXTURES_DIR / "data.txt"
        bad_file.write_text("hello")
        try:
            with pytest.raises(ValueError, match="Unsupported source type"):
                load_documents(bad_file)
        finally:
            bad_file.unlink(missing_ok=True)

    def test_missing_file_raises(self):
        missing = FIXTURES_DIR / "nonexistent.pdf"
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_documents(missing)

    def test_directory_raises(self):
        FIXTURES_DIR.mkdir(exist_ok=True)
        pdf_dir = FIXTURES_DIR / "fake.pdf"
        pdf_dir.mkdir(exist_ok=True)
        try:
            with pytest.raises(ValueError, match="Path is not a file"):
                load_documents(pdf_dir)
        finally:
            pdf_dir.rmdir()

    def test_empty_source_raises(self):
        with pytest.raises(ValueError, match="Source cannot be empty"):
            load_documents("")

    def test_whitespace_only_source_raises(self):
        with pytest.raises(ValueError, match="Source cannot be empty"):
            load_documents("   ")
