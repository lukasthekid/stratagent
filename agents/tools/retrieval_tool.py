import os
from pathlib import Path

from crewai.tools import BaseTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config import settings
from retrieval import retrieve_with_rerank


class RetrievalInput(BaseModel):
    query: str = Field(description="Search query")


def _format_source_label(doc: Document) -> str:
    """Build a citation-friendly label from document metadata."""
    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page")
    page_label = doc.metadata.get("page_label")
    title = doc.metadata.get("title")

    # Use title as primary label for web docs when available
    if title and source.startswith(("http://", "https://")):
        label = title
    else:
        # For file paths, use basename for cleaner display
        label = Path(source).name if os.path.sep in str(source) else source

    # Add page reference for PDFs (1-based for display)
    if page is not None:
        display_page = page_label if page_label is not None else int(page) + 1
        label = f"{label} (p. {display_page})"
    elif page_label is not None:
        label = f"{label} (p. {page_label})"

    return label


class RetrievalTool(BaseTool):
    name: str = "Document Retrieval Tool"
    description: str = "Search internal docs (10-K, earnings, reports). Use before claiming financials."
    args_schema: type[BaseModel] = RetrievalInput

    def _run(self, query: str) -> str:
        try:
            results: list[Document] = retrieve_with_rerank.invoke(
                {
                    "query": query,
                    "retrieval_k": settings.retrieval_top_k,
                    "rerank_k": settings.rerank_top_k,
                }
            )
        except Exception as e:
            return f"Retrieval error: {e}. Try rephrasing your query or check that the document database is populated."

        if not results:
            return "No relevant documents found. Try broadening your query."

        formatted_results = []
        max_chunk_len = 300
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            label = _format_source_label(doc)
            content = doc.page_content
            if len(content) > max_chunk_len:
                content = content[:max_chunk_len] + "..."
            # Include both Source (for URL/path extraction) and Label (for display)
            formatted_results.append(f"[{i}] Source: {source}\nLabel: {label}\n{content}")

        return "\n---\n".join(formatted_results)