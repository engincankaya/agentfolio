import json

from langchain_core.tools import tool
from pydantic import BaseModel

from src.core.logging import logger


def build_search_tool(rag_service=None):
    """
    Returns a LangChain tool that performs semantic search against the portfolio vector store.
    Accepts an optional RAGService instance; falls back to the module-level default.
    """
    from src.services.rag_service import RAGService

    _service = rag_service or RAGService()

    @tool("search_portfolio")
    def search_portfolio(query: str, k: int = 4) -> str:
        """
        Performs semantic search in the portfolio database (resume, experience, projects, tech stack).
        Use this tool whenever you need specific information about the portfolio owner's past work, skills or biography.
        """
        logger.info(f"Searching portfolio for: '{query}'")
        results = _service.similarity_search(query, k)

        if not results:
            return "No relevant portfolio information found."

        formatted_results = []
        for doc, score in results:
            category = doc.metadata.get("category", "unknown")
            filename = doc.metadata.get("filename", "?")
            formatted_results.append(
                f"--- Category: {category} | File: {filename} ---\n{doc.page_content}"
            )

        return "\n\n".join(formatted_results)

    return search_portfolio


# --- Handoff Tools (parameterless signal tools) ---

class _HandoffInput(BaseModel):
    """No parameters needed for handoff tools."""
    pass


@tool("handoff_to_github", args_schema=_HandoffInput)
def handoff_to_github() -> str:
    """
    Transfers control to GitHub Agent for GitHub-related questions
    (repositories, issues, pull requests, actions, commits, etc.).
    TRANSFERS CONTROL SILENTLY - do not generate any text before calling this.
    """
    return json.dumps({"status": "handoff_initiated", "target": "github_agent"})


@tool("handoff_to_calendar", args_schema=_HandoffInput)
def handoff_to_calendar() -> str:
    """
    Transfers control to Calendar Agent for Google Calendar questions
    (events, meetings, scheduling, availability, etc.).
    TRANSFERS CONTROL SILENTLY - do not generate any text before calling this.
    """
    return json.dumps({"status": "handoff_initiated", "target": "calendar_agent"})
