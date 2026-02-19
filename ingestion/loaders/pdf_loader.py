"""PDF document loader using pdfplumber."""

from datetime import datetime
from pathlib import Path

import pdfplumber

from ingestion.models import Document, DocumentMetadata


class PDFLoader:
    """Load PDF documents (annual reports, strategy docs) into standardized Documents."""

    def __init__(
        self,
        path: str | Path,
        *,
        company_name: str | None = None,
        source_date: datetime | None = None,
    ) -> None:
        """Initialize PDF loader.

        Args:
            path: Path to the PDF file.
            company_name: Optional company name for metadata.
            source_date: Optional document date (defaults to file mtime).
        """
        self.path = Path(path)
        self.company_name = company_name
        self.source_date = source_date

    def load(self) -> list[Document]:
        """Extract text from PDF and return as Documents (one per page)."""
        if not self.path.exists():
            raise FileNotFoundError(f"PDF not found: {self.path}")

        date = self.source_date
        if date is None:
            try:
                mtime = self.path.stat().st_mtime
                date = datetime.fromtimestamp(mtime)
            except OSError:
                date = None

        documents: list[Document] = []

        with pdfplumber.open(self.path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                content = text if text else ""

                metadata = DocumentMetadata(
                    source=str(self.path),
                    date=date,
                    company_name=self.company_name,
                    extra={"page": i + 1, "total_pages": len(pdf.pages)},
                )
                documents.append(Document(content=content, metadata=metadata))

        return documents
