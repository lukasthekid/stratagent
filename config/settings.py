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
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    groq_api_key: str = "grq_xxxxxxx"
    pinecone_api_key: str = "pc_xxxxxxx"
    pinecone_index_name: str = "stratagent"
    embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2"
    # Upsert tuning: embedding batch (HuggingFace), Pinecone upsert batch, pool threads
    embedding_batch_size: int = 64
    upsert_batch_size: int = 64
    pinecone_pool_threads: int = 8

    # CORS (comma-separated list, e.g. "http://localhost:3000,https://app.example.com")
    cors_origins: str = "*"

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
