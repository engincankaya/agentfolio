"""CLI script to ingest markdown documents into the Qdrant vector store."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.rag_service import RAGService
from src.core.logging import logger


def main():
    logger.info("Starting document ingestion...")
    try:
        result = RAGService().ingest_documents()
        logger.info(
            f"Ingestion complete: {result['documents_processed']} documents, "
            f"{result['chunks_created']} chunks indexed."
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
