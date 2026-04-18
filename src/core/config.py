import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    embedding_model: str = "openai/text-embedding-3-small"
    llm_model: str = "google/gemini-3.1-flash-lite-preview"

    # Qdrant
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = str(BASE_DIR / "data" / "vector_db")
    qdrant_collection: str = "portfolio"

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4

    # Data
    raw_data_path: str = str(BASE_DIR / "data" / "raw")

    # GitHub MCP
    github_pat: str = ""

    # Google Calendar MCP
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""

    # Mindmap MCP
    mindmap_mcp_server_path: str = ""

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_tracing: str = "false"
    langsmith_project: str = "agenticfolio"

    def apply_langsmith_env(self) -> None:
        """Explicitly set LANGSMITH env vars so the LangChain SDK picks them up."""
        os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint
        os.environ["LANGSMITH_TRACING"] = self.langsmith_tracing
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project


settings = Settings()
settings.apply_langsmith_env()
