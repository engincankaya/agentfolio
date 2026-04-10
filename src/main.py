from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.api.routes import router
from src.core.config import settings
from src.core.logging import logger
from src.services.rag_service import RAGService
from src.services.embedding_service import EmbeddingService
from src.agents.graph import create_portfolio_graph
from src.tools.portfolio_tools import build_search_tool


def _build_mcp_config() -> dict:
    """Builds MCP server configuration from settings."""
    servers = {}

    if settings.github_pat:
        servers["github"] = {
            "command": "docker",
            "args": [
                "run", "-i", "--rm",
                "-e", "GITHUB_PERSONAL_ACCESS_TOKEN",
                "ghcr.io/github/github-mcp-server",
                "stdio",
                "--tools",
                "get_me,search_repositories,get_file_contents,list_commits,"
                "list_branches,list_tags,list_releases,get_latest_release,"
                "get_commit,get_tag,get_release_by_tag,search_code",
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_pat,
            },
            "transport": "stdio",
        }

    if settings.google_oauth_client_id and settings.google_oauth_client_secret:
        servers["google_calendar"] = {
            "command": "uvx",
            "args": ["workspace-mcp", "--tools", "calendar"],
            "env": {
                "GOOGLE_OAUTH_CLIENT_ID": settings.google_oauth_client_id,
                "GOOGLE_OAUTH_CLIENT_SECRET": settings.google_oauth_client_secret,
            },
            "transport": "stdio",
        }

    return servers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes shared services on startup and cleans up on shutdown."""
    logger.info("Starting Agenticfolio API server...")

    embedding_service = EmbeddingService(cfg=settings)
    rag_service = RAGService(cfg=settings, embedding_service=embedding_service)
    search_tool = build_search_tool(rag_service=rag_service)

    mcp_config = _build_mcp_config()
    github_tools = []
    calendar_tools = []

    if mcp_config:
        try:
            mcp_client = MultiServerMCPClient(mcp_config)

            if "github" in mcp_config:
                github_tools = await mcp_client.get_tools(server_name="github")
                get_me_tool = next((t for t in github_tools if t.name == "get_me"), None)
                if get_me_tool:
                    app.state.github_user_context = await get_me_tool.ainvoke({})
                    github_tools = [t for t in github_tools if t.name != "get_me"]
                    logger.info(f"GitHub authenticated user resolved.")

            # if "google_calendar" in mcp_config:
            #     calendar_tools = await mcp_client.get_tools(server_name="google_calendar")

            logger.info(
                f"MCP tools loaded — GitHub: {len(github_tools)}, Calendar: {len(calendar_tools)}"
            )
        except Exception:
            logger.exception("Failed to load MCP tools.")
            github_tools = []
            calendar_tools = []
    else:
        logger.warning(
            "No MCP servers configured. GitHub and Calendar agents will have no tools."
        )

    graph = create_portfolio_graph(
        search_tool=search_tool,
        github_tools=github_tools,
        calendar_tools=calendar_tools,
    )

    app.state.rag_service = rag_service
    app.state.graph = graph

    yield

    logger.info("Agenticfolio API server shutting down.")


app = FastAPI(
    title="Agenticfolio",
    description="AI-powered portfolio digital twin with RAG and multi-agent architecture",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "project": settings.langsmith_project}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
