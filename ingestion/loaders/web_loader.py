"""Web content document loader using BeautifulSoup."""

from datetime import datetime
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from ingestion.models import Document, DocumentMetadata


class WebLoader:
    """Load web pages into standardized Documents."""

    def __init__(
        self,
        url: str,
        *,
        company_name: str | None = None,
        source_date: datetime | None = None,
        timeout: float = 30.0,
        user_agent: str = "Stratagent/1.0",
    ) -> None:
        """Initialize web loader.

        Args:
            url: URL to fetch.
            company_name: Optional company name for metadata.
            source_date: Optional document date (defaults to fetch time).
            timeout: Request timeout in seconds.
            user_agent: User-Agent header for requests.
        """
        self.url = url
        self.company_name = company_name
        self.source_date = source_date
        self.timeout = timeout
        self.user_agent = user_agent

    def load(self) -> list[Document]:
        """Fetch URL, extract main text, and return as Document."""
        with httpx.Client(
            timeout=self.timeout,
            headers={"User-Agent": self.user_agent},
            follow_redirects=True,
        ) as client:
            response = client.get(self.url)
            response.raise_for_status()

        date = self.source_date or datetime.now()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        content = "\n".join(line for line in text.splitlines() if line.strip())

        metadata = DocumentMetadata(
            source=self.url,
            date=date,
            company_name=self.company_name,
            extra={
                "domain": urlparse(self.url).netloc,
                "title": soup.title.string if soup.title else None,
            },
        )
        return [Document(content=content, metadata=metadata)]
