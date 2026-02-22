from crewai.tools import BaseTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config import settings
from retrieval import retrieve_with_rerank


class RetrievalInput(BaseModel):
    query: str = Field(description="The search query to find relevant documents")


class RetrievalTool(BaseTool):
    name: str = "Document Retrieval Tool"
    description: str = """
    Use this tool to search and retrieve relevant information from the internal document database.
    This includes annual reports, 10-K filings, earnings calls, and industry reports.
    Always use this before making any claims about a company's financials or strategy.
    Input a specific, focused query for best results.
    """
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
        for i, doc in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            formatted_results.append(f"[{i}] Source: {source}\n{doc.page_content}")

        return "\n---\n".join(formatted_results)