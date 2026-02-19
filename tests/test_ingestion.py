"""Sophisticated tests for ingestion module."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestion import Document, DocumentMetadata, load_documents
from ingestion.loaders import CSVLoader, PDFLoader, WebLoader
from ingestion.schemas import FinancialRow


# --- Document & Metadata Model Tests ---


class TestDocumentMetadata:
    """Tests for DocumentMetadata model."""

    def test_minimal_metadata(self) -> None:
        """Metadata requires only source."""
        meta = DocumentMetadata(source="file.pdf")
        assert meta.source == "file.pdf"
        assert meta.date is None
        assert meta.company_name is None
        assert meta.extra == {}

    def test_full_metadata(self) -> None:
        """Metadata accepts all fields."""
        dt = datetime(2024, 1, 15)
        meta = DocumentMetadata(
            source="https://example.com",
            date=dt,
            company_name="Acme Corp",
            extra={"page": 1, "domain": "example.com"},
        )
        assert meta.source == "https://example.com"
        assert meta.date == dt
        assert meta.company_name == "Acme Corp"
        assert meta.extra["page"] == 1

    def test_metadata_serialization_roundtrip(self) -> None:
        """Metadata serializes and deserializes correctly."""
        meta = DocumentMetadata(
            source="report.pdf",
            date=datetime(2024, 6, 1),
            company_name="TechCo",
        )
        dumped = meta.model_dump()
        restored = DocumentMetadata.model_validate(dumped)
        assert restored == meta


class TestDocument:
    """Tests for Document model."""

    def test_document_creation(self) -> None:
        """Document requires content and metadata."""
        meta = DocumentMetadata(source="test.pdf", company_name="Corp")
        doc = Document(content="Hello World", metadata=meta)
        assert doc.content == "Hello World"
        assert doc.metadata.source == "test.pdf"

    def test_document_with_empty_content(self) -> None:
        """Document allows empty content (e.g. blank PDF page)."""
        meta = DocumentMetadata(source="blank.pdf")
        doc = Document(content="", metadata=meta)
        assert doc.content == ""
        assert doc.metadata.source == "blank.pdf"

    def test_document_json_serialization(self) -> None:
        """Document serializes to JSON for storage/indexing."""
        meta = DocumentMetadata(source="x.pdf", company_name="X")
        doc = Document(content="Data", metadata=meta)
        json_str = doc.model_dump_json()
        assert "Data" in json_str
        assert "x.pdf" in json_str
        restored = Document.model_validate_json(json_str)
        assert restored == doc


# --- PDF Loader Tests ---


class TestPDFLoader:
    """Tests for PDFLoader."""

    def test_load_pdf_extracts_text(self, sample_pdf_path: Path) -> None:
        """PDFLoader extracts text from each page."""
        loader = PDFLoader(sample_pdf_path, company_name="Acme Corp")
        docs = loader.load()
        assert len(docs) == 2
        assert "Annual Report 2024" in docs[0].content
        assert "Acme Corporation" in docs[0].content
        assert "Page 2" in docs[1].content
        assert "Strategic outlook" in docs[1].content

    def test_load_pdf_metadata(self, sample_pdf_path: Path) -> None:
        """PDF documents have correct metadata."""
        loader = PDFLoader(sample_pdf_path, company_name="Acme")
        docs = loader.load()
        for doc in docs:
            assert doc.metadata.source == str(sample_pdf_path)
            assert doc.metadata.company_name == "Acme"
            assert "page" in doc.metadata.extra
            assert "total_pages" in doc.metadata.extra
            assert doc.metadata.extra["total_pages"] == 2

    def test_load_pdf_custom_source_date(self, sample_pdf_path: Path) -> None:
        """PDFLoader uses provided source_date when given."""
        dt = datetime(2023, 12, 25)
        loader = PDFLoader(sample_pdf_path, source_date=dt)
        docs = loader.load()
        assert docs[0].metadata.date == dt

    def test_load_pdf_file_not_found(self) -> None:
        """PDFLoader raises FileNotFoundError for missing file."""
        loader = PDFLoader("/nonexistent/path/report.pdf")
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            loader.load()

    def test_load_pdf_accepts_str_path(self, sample_pdf_path: Path) -> None:
        """PDFLoader accepts string path."""
        loader = PDFLoader(str(sample_pdf_path))
        docs = loader.load()
        assert len(docs) >= 1


# --- CSV Loader Tests ---


class TestCSVLoader:
    """Tests for CSVLoader."""

    def test_load_csv_basic(self, sample_csv_path: Path) -> None:
        """CSVLoader loads tabular data as Document."""
        loader = CSVLoader(sample_csv_path, company_name="Acme")
        docs = loader.load()
        assert len(docs) == 1
        assert "2024-01-01" in docs[0].content
        assert "1000000" in docs[0].content
        assert docs[0].metadata.extra["rows"] == 3
        assert "revenue" in docs[0].metadata.extra["columns"]

    def test_load_csv_with_schema_validation(self, sample_csv_path: Path) -> None:
        """CSVLoader validates rows when schema provided."""
        loader = CSVLoader(sample_csv_path, schema=FinancialRow)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].metadata.extra["rows"] == 3

    def test_load_csv_schema_validation_fails(
        self, sample_csv_with_nulls: Path, invalid_csv_schema: type
    ) -> None:
        """CSVLoader raises ValueError when schema validation fails."""
        loader = CSVLoader(sample_csv_with_nulls, schema=invalid_csv_schema)
        with pytest.raises(ValueError, match="Schema validation failed"):
            loader.load()

    def test_load_csv_custom_encoding(self, tmp_path: Path) -> None:
        """CSVLoader respects encoding parameter."""
        csv_path = tmp_path / "latin1.csv"
        csv_path.write_text(
            "col1,col2\ncafé,naïve\n",
            encoding="latin-1",
        )
        loader = CSVLoader(csv_path, encoding="latin-1")
        docs = loader.load()
        assert "café" in docs[0].content or "caf" in docs[0].content

    def test_load_csv_file_not_found(self) -> None:
        """CSVLoader raises FileNotFoundError for missing file."""
        loader = CSVLoader("/nonexistent/data.csv")
        with pytest.raises(FileNotFoundError, match="CSV not found"):
            loader.load()

    def test_load_csv_empty_file(self, tmp_path: Path) -> None:
        """CSVLoader handles empty CSV (headers only)."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("a,b,c\n")
        loader = CSVLoader(csv_path)
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].metadata.extra["rows"] == 0


# --- Web Loader Tests ---


class TestWebLoader:
    """Tests for WebLoader."""

    def test_load_web_extracts_text(self) -> None:
        """WebLoader extracts main text, strips scripts/styles."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.text = """
        <html><head><title>Report</title>
        <script>alert('x')</script><style>body{}</style></head>
        <body><nav>Skip</nav><p>Main content here</p><footer>Skip</footer></body>
        </html>
        """
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("ingestion.loaders.web_loader.httpx.Client") as mock_httpx:
            mock_httpx.return_value.__enter__.return_value = mock_client

            loader = WebLoader("https://example.com/report", company_name="Example")
            docs = loader.load()

        assert len(docs) == 1
        assert "Main content here" in docs[0].content
        assert "alert" not in docs[0].content
        assert "Skip" not in docs[0].content

    def test_load_web_metadata(self) -> None:
        """Web Document has correct metadata with domain and title."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.text = "<html><head><title>Q4 Report</title></head><body><p>x</p></body></html>"
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("ingestion.loaders.web_loader.httpx.Client") as mock_httpx:
            mock_httpx.return_value.__enter__.return_value = mock_client

            loader = WebLoader("https://acme.com/docs/report")
            docs = loader.load()

        assert docs[0].metadata.source == "https://acme.com/docs/report"
        assert docs[0].metadata.extra["domain"] == "acme.com"
        assert docs[0].metadata.extra["title"] == "Q4 Report"

    def test_load_web_uses_custom_source_date(self) -> None:
        """WebLoader uses provided source_date."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.text = "<html><body><p>x</p></body></html>"
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        dt = datetime(2024, 1, 1)

        with patch("ingestion.loaders.web_loader.httpx.Client") as mock_httpx:
            mock_httpx.return_value.__enter__.return_value = mock_client

            loader = WebLoader("https://example.com", source_date=dt)
            docs = loader.load()

        assert docs[0].metadata.date == dt

    def test_load_web_http_error(self) -> None:
        """WebLoader propagates HTTP errors."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("ingestion.loaders.web_loader.httpx.Client") as mock_httpx:
            mock_httpx.return_value.__enter__.return_value = mock_client

            loader = WebLoader("https://example.com/404")
            with pytest.raises(Exception, match="404"):
                loader.load()


# --- Unified load_documents Tests ---


class TestLoadDocuments:
    """Tests for load_documents unified interface."""

    def test_dispatch_unsupported_extension(self) -> None:
        """Unsupported file extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported source type"):
            load_documents("/path/to/file.txt")

    def test_dispatch_unsupported_no_extension(self) -> None:
        """Path with no extension raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported source type"):
            load_documents("/path/to/file")

    def test_dispatch_csv(self, sample_csv_path: Path) -> None:
        """load_documents dispatches CSV to CSVLoader."""
        docs = load_documents(sample_csv_path, company_name="Acme Corp")
        assert len(docs) == 1
        assert docs[0].metadata.company_name == "Acme Corp"
        assert "revenue" in docs[0].content

    def test_dispatch_csv_with_schema(self, sample_csv_path: Path) -> None:
        """load_documents passes csv_schema to CSVLoader."""
        docs = load_documents(
            sample_csv_path,
            company_name="Acme",
            csv_schema=FinancialRow,
        )
        assert len(docs) == 1
        assert docs[0].metadata.extra["rows"] == 3

    def test_dispatch_pdf(self, sample_pdf_path: Path) -> None:
        """load_documents dispatches PDF to PDFLoader."""
        docs = load_documents(sample_pdf_path, company_name="Acme")
        assert len(docs) == 2
        assert docs[0].metadata.company_name == "Acme"
        assert "Annual Report" in docs[0].content

    def test_dispatch_web_url(self) -> None:
        """load_documents dispatches URL to WebLoader."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.text = "<html><body><p>Web content</p></body></html>"
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("ingestion.loaders.web_loader.httpx.Client") as mock_httpx:
            mock_httpx.return_value.__enter__.return_value = mock_client

            docs = load_documents("https://example.com/page", company_name="Example")
        assert len(docs) == 1
        assert "Web content" in docs[0].content

    def test_dispatch_http_url(self) -> None:
        """load_documents accepts http:// URLs."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.text = "<html><body><p>x</p></body></html>"
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("ingestion.loaders.web_loader.httpx.Client") as mock_httpx:
            mock_httpx.return_value.__enter__.return_value = mock_client

            docs = load_documents("http://example.com")
        assert len(docs) == 1

    def test_dispatch_pdf_uppercase_extension(self, sample_pdf_path: Path, tmp_path: Path) -> None:
        """load_documents handles .PDF (uppercase) extension."""
        pdf_upper = tmp_path / "report.PDF"
        pdf_upper.write_bytes(sample_pdf_path.read_bytes())
        docs = load_documents(pdf_upper)
        assert len(docs) >= 1

    def test_dispatch_csv_uppercase_extension(self, sample_csv_path: Path, tmp_path: Path) -> None:
        """load_documents handles .CSV (uppercase) extension."""
        csv_upper = tmp_path / "data.CSV"
        csv_upper.write_text(sample_csv_path.read_text())
        docs = load_documents(csv_upper)
        assert len(docs) == 1

    def test_dispatch_https_and_http_urls(self) -> None:
        """load_documents accepts both http:// and https:// URLs."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.text = "<html><body><p>x</p></body></html>"
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        with patch("ingestion.loaders.web_loader.httpx.Client") as m:
            m.return_value.__enter__.return_value = mock_client
            docs_https = load_documents("https://example.com")
            docs_http = load_documents("HTTP://example.com")
        assert len(docs_https) == 1
        assert len(docs_http) == 1
