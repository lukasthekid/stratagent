from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from config import settings


class SearchInput(BaseModel):
    query: str = Field(description="Web search query")


class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = "Search web for news, competitors, market data. Use Document Retrieval for historical financials."
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        try:
            client = TavilyClient(api_key=settings.tavily_api_key)
            results = client.search(query=query, max_results=3, search_depth="advanced")
        except Exception as e:
            return f"Web search error: {e}. Check your Tavily API key and network connection."

        formatted = []
        max_content_len = 200
        for r in results.get("results", []):
            content = r.get("content", "")[:max_content_len]
            if len(r.get("content", "")) > max_content_len:
                content += "..."
            formatted.append(f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', '')}\nContent: {content}\n")

        return "\n---\n".join(formatted) if formatted else "No results found."