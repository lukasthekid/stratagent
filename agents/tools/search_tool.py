from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from config import settings


class SearchInput(BaseModel):
    query: str = Field(description="Web search query for current market information")


class WebSearchTool(BaseTool):
    name: str = "Web Search Tool"
    description: str = """
    Use this tool to search the web for current news, recent events, competitor information,
    and market data that may not be in the document database. 
    Use for: recent news, current stock data, competitor moves, industry trends.
    Do NOT use for historical financials — use the Document Retrieval Tool for those.
    """
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        try:
            client = TavilyClient(api_key=settings.tavily_api_key)
            results = client.search(query=query, max_results=5, search_depth="advanced")
        except Exception as e:
            return f"Web search error: {e}. Check your Tavily API key and network connection."

        formatted = []
        for r in results.get("results", []):
            formatted.append(f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', '')}\nContent: {r.get('content', '')}\n")

        return "\n---\n".join(formatted) if formatted else "No results found."