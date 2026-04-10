from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from src.core.config import Settings, settings
from src.core.logging import logger
from src.services.embedding_service import EmbeddingService


_CATEGORY_MAP: dict[str, str] = {
    "about": "bio",
    "bio": "bio",
    "experience": "experience",
    "project": "project",
    "closync": "project",
    "dental": "project",
    "gateway": "project",
}


class RAGService:
    """Handles document ingestion and semantic search against the vector store."""

    def __init__(
        self,
        cfg: Settings = settings,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self._cfg = cfg
        self._embedding_service = embedding_service or EmbeddingService(cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_documents(self) -> dict:
        """Loads, chunks, and indexes all Markdown files into Qdrant."""
        raw_path = Path(self._cfg.raw_data_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"Data directory not found: {raw_path}")

        documents = self._load_documents(raw_path)
        self._attach_metadata(documents)
        chunks = self._split(documents)
        self._index(chunks)

        return {
            "documents_processed": len(documents),
            "chunks_created": len(chunks),
        }

    def similarity_search(self, query: str, k: int | None = None) -> list:
        """Returns the top-k most relevant document chunks for a query."""
        k = k or self._cfg.retrieval_k
        store = self._get_vector_store()
        results = store.similarity_search_with_score(query, k=k)
        logger.info(f"Found {len(results)} results for query: '{query[:50]}...'")
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_documents(self, raw_path: Path) -> list:
        loader = DirectoryLoader(
            str(raw_path),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {raw_path}")
        return documents

    def _attach_metadata(self, documents: list) -> None:
        for doc in documents:
            filename = Path(doc.metadata.get("source", "")).name
            doc.metadata["filename"] = filename
            doc.metadata["category"] = self._detect_category(filename)

    def _split(self, documents: list) -> list:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._cfg.chunk_size,
            chunk_overlap=self._cfg.chunk_overlap,
            separators=["\n## ", "\n### ", "\n- ", "\n\n", "\n", " "],
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunks")
        return chunks

    def _index(self, chunks: list) -> None:
        embeddings = self._embedding_service.get()
        client = QdrantClient(path=self._cfg.qdrant_path)
        collection_name = self._cfg.qdrant_collection

        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        store.add_documents(chunks)
        logger.info(f"Indexed {len(chunks)} chunks into '{collection_name}'")

    def _get_vector_store(self) -> QdrantVectorStore:
        client = QdrantClient(path=self._cfg.qdrant_path)
        return QdrantVectorStore(
            client=client,
            collection_name=self._cfg.qdrant_collection,
            embedding=self._embedding_service.get(),
        )

    @staticmethod
    def _detect_category(filename: str) -> str:
        name = filename.lower().replace(".md", "")
        for key, category in _CATEGORY_MAP.items():
            if key in name:
                return category
        return "general"
