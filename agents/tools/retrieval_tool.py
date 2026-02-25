from crewai.tools import BaseTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config import settings
from retrieval import retrieve_with_rerank


class RetrievalInput(BaseModel):
    query: str = Field(description="Search query")


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
            content = doc.page_content
            if len(content) > max_chunk_len:
                content = content[:max_chunk_len] + "..."
            formatted_results.append(f"[{i}] Source: {source}\n{content}")

        return "\n---\n".join(formatted_results)