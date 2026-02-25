"""Application settings loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env path relative to project root (parent of config/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application configuration. All values loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    api_host: str = "localhost"
    api_port: int = 8000
    api_reload: bool = True

    groq_api_key: str = "grq_xxxxxxx"
    pinecone_api_key: str = "pc_xxxxxxx"
    gemini_api_key: str = "gemini_xxxxxxx"
    pinecone_index_name: str = "stratagent"
    tavily_api_key: str = "tvly-xxxxxxxxx"
    embedding_model: str = "llama-text-embed-v2"
    embedding_dimensions: int = 1024
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    llm_model: str = "groq/groq/compound"
    # Retrieval tuning: vector search candidates, final reranked count
    retrieval_top_k: int = 12
    rerank_top_k: int = 3
    retriever_threshold: float = 0.0
    reranker_threshold: float = 0.0
    # Chunk size for document ingestion (characters)
    chunk_size: int = 384
    max_tokens: int = 2048
    # Upsert tuning: embedding batch (HuggingFace), Pinecone upsert batch, pool threads
    embedding_batch_size: int = 64
    upsert_batch_size: int = 64
    pinecone_pool_threads: int = 8

    # CORS (comma-separated list, e.g. "http://localhost:3000,https://app.example.com")
    cors_origins: str = "*"
    # MLFLow settings
    mlflow_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "CrewAI - StratAgent"

    # Rate limit retry: max attempts, default wait when API doesn't specify
    llm_rate_limit_max_retries: int = 5
    llm_rate_limit_default_wait_seconds: float = 5.0

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins as list. '*' returns ['*']."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # Environment
    env: str = "development"

    # Frontend (for Streamlit)
    api_url: str = "http://localhost:8000"

    user_agent: str = "Stratagent stratagent@example.com"


settings = Settings()
