"""CLI script to ingest markdown documents into Qdrant vector store."""
import sys
from pathlib import Path
from textwrap import shorten

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.rag_service import RAGService
from src.core.logging import logger

def pretty_print_results(results: list) -> None:
    if not results:
        print("\nNo results found.")
        return

    print(f"\nFound {len(results)} result(s)\n")

    for i, (doc,score) in enumerate(results, 1):
        category = doc.metadata.get("category", "-")
        project_name = doc.metadata.get("project_name", "-")
        company_name = doc.metadata.get("company_name", "-")
        source = doc.metadata.get("source", "-")
        
        content = shorten(
            doc.page_content.replace("\n", " "),
            width=400,
            placeholder="..."
        )

        print(f"[{i}] {project_name}  |  {company_name}  |  {category}")
        print(f"\n[{i}] score={score:.4f}")
        print(f"    Source : {source}")
        print(f"    Content: {content}")
        print()

def main():
    logger.info("Starting document ingestion...")
    try:
        result = RAGService().similarity_search(query="Has he worked on multi-agent systems?",k=3)
        pretty_print_results(result)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
