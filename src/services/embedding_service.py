from langchain_openai import OpenAIEmbeddings

from src.core.config import Settings, settings


class EmbeddingService:
    """Manages a singleton OpenAIEmbeddings instance."""

    _instance: OpenAIEmbeddings | None = None

    def __init__(self, cfg: Settings = settings) -> None:
        self._cfg = cfg

    def get(self) -> OpenAIEmbeddings:
        """Returns the shared embedding model, creating it on first call."""
        if EmbeddingService._instance is None:
            EmbeddingService._instance = OpenAIEmbeddings(
                model=self._cfg.embedding_model,
                openai_api_key=self._cfg.openrouter_api_key,
                openai_api_base=self._cfg.openrouter_base_url,
            )
        return EmbeddingService._instance

    @classmethod
    def reset(cls) -> None:
        """Clears the cached instance (useful in tests)."""
        cls._instance = None
