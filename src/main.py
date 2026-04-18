from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.api.routes import router
from src.core.config import settings
from src.core.logging import logger
from src.services.rag_service import RAGService
from src.services.embedding_service import EmbeddingService
from src.services.github_catalog import extract_repository_items, normalize_github_user_context
from src.services.portfolio_catalog import load_private_portfolio_catalog
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
                "get_me,get_file_contents,search_repositories"
            ],
            "env": {
                "GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_pat,
            },
            "transport": "stdio",
        }

    if settings.mindmap_mcp_server_path:
        servers["mindmap"] = {
            "command": "node",
            "args": [settings.mindmap_mcp_server_path],
            "transport": "stdio",
        }

    return servers


async def _build_public_repo_catalog(github_tools: list, github_user_context) -> list[dict]:
    normalized_context = normalize_github_user_context(github_user_context)
    login = normalized_context.get("login", "")

    search_tool = next((tool for tool in github_tools if tool.name == "search_repositories"), None)
    if not login or not search_tool:
        return []

    supported_args = getattr(search_tool, "args", {}) or {}
    invoke_payload = {}
    query_value = f"user:{login} sort:updated-desc"
    if not supported_args or "query" in supported_args:
        invoke_payload["query"] = query_value
    elif "q" in supported_args:
        invoke_payload["q"] = query_value

    if not supported_args or "perPage" in supported_args:
        invoke_payload["perPage"] = 10
    elif "per_page" in supported_args:
        invoke_payload["per_page"] = 10

    if not supported_args or "page" in supported_args:
        invoke_payload["page"] = 1

    try:
        result = await search_tool.ainvoke(invoke_payload)
    except Exception:
        logger.exception("Failed to build public repo catalog from GitHub search.")
        return []

    items = extract_repository_items(result)

    catalog = []
    for repo in items[:10]:
        if not isinstance(repo, dict):
            continue
        owner = ""
        if isinstance(repo.get("owner"), dict):
            owner = repo["owner"].get("login") or repo["owner"].get("name") or ""
        full_name = repo.get("full_name") or ""
        if not full_name and owner and repo.get("name"):
            full_name = f"{owner}/{repo['name']}"
        catalog.append(
            {
                "name": repo.get("name") or repo.get("full_name") or "",
                "full_name": full_name,
                "description": repo.get("description") or "",
                "language": repo.get("language") or "",
                "updated_at": repo.get("updated_at") or repo.get("pushed_at") or "",
            }
        )

    return catalog


def _format_assistant_context(public_repo_catalog: list[dict], private_portfolio_catalog: list[dict]) -> str:
    public_lines = [
        f"- {repo['full_name'] or repo['name']} | lang={repo['language'] or '?'} | updated={repo['updated_at'] or '?'} | {repo['description'] or 'No description'}"
        for repo in public_repo_catalog
    ] or ["- No public repository catalog available."]
    private_lines = [
        (
            f"- project={entry['project_name']} | company={entry['company_name'] or '?'}"
            f" | role={entry.get('role') or '?'} | period={entry.get('period') or '?'}"
            f" | employment_type={entry.get('employment_type') or '?'}"
            f" | location={entry.get('location') or '?'}"
            f" | visibility={entry['visibility']} | {entry['summary']}"
        )
        for entry in private_portfolio_catalog
    ] or ["- No private portfolio catalog available."]

    return (
        "Assistant routing context:\n"
        "Public repo catalog:\n"
        + "\n".join(public_lines)
        + "\n\nPrivate portfolio catalog:\n"
        + "\n".join(private_lines)
    )


def _format_specialist_context(public_repo_catalog: list[dict]) -> str:
    public_lines = [
        f"- {repo['full_name'] or repo['name']} | lang={repo['language'] or '?'} | updated={repo['updated_at'] or '?'} | {repo['description'] or 'No description'}"
        for repo in public_repo_catalog
    ] or ["- No public repository catalog available."]

    return (
        "Specialist repository context:\n"
        "Prefer these repositories as candidates when the conversation implies a public repo:\n"
        + "\n".join(public_lines)
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes shared services on startup and cleans up on shutdown."""
    logger.info("Starting Agenticfolio API server...")

    embedding_service = EmbeddingService(cfg=settings)
    rag_service = RAGService(cfg=settings, embedding_service=embedding_service)
    search_tool = build_search_tool(rag_service=rag_service)
    private_portfolio_catalog = load_private_portfolio_catalog(settings.raw_data_path)

    mcp_config = _build_mcp_config()
    specialist_tools = []
    public_repo_catalog = []

    if mcp_config:
        try:
            mcp_client = MultiServerMCPClient(mcp_config)

            if "github" in mcp_config:
                github_tools = await mcp_client.get_tools(server_name="github")
                get_me_tool = next((t for t in github_tools if t.name == "get_me"), None)
                if get_me_tool:
                    raw_github_user_context = await get_me_tool.ainvoke({})
                    app.state.github_user_context = normalize_github_user_context(raw_github_user_context)
                    github_tools = [t for t in github_tools if t.name != "get_me"]
                    public_repo_catalog = await _build_public_repo_catalog(
                        github_tools,
                        app.state.github_user_context,
                    )
                    logger.info(f"GitHub authenticated user resolved.")
                get_file_contents_tool = next(
                    (t for t in github_tools if t.name == "get_file_contents"), None
                )
                if get_file_contents_tool:
                    specialist_tools.append(get_file_contents_tool)

            if "mindmap" in mcp_config:
                logger.info("Connecting to Mindmap MCP server...")
                all_mindmap_tools = await mcp_client.get_tools(server_name="mindmap")
                logger.info(
                    f"Mindmap MCP server connected — {len(all_mindmap_tools)} tools available: "
                    f"{[t.name for t in all_mindmap_tools]}"
                )
                mindmap_tools = [
                    t for t in all_mindmap_tools
                    if t.name in ("mindmap.overview", "mindmap.find")
                ]
                specialist_tools.extend(mindmap_tools)
                logger.info(f"Mindmap tools filtered for specialist: {[t.name for t in mindmap_tools]}")

            logger.info(
                f"MCP tools loaded — Specialist: {len(specialist_tools)}, Public repos: {len(public_repo_catalog)}"
            )
        except Exception:
            logger.exception("Failed to load MCP tools.")
            specialist_tools = []
            public_repo_catalog = []
    else:
        logger.warning("No MCP servers configured. Specialist node will have no external tools.")

    assistant_context = _format_assistant_context(
        public_repo_catalog=public_repo_catalog,
        private_portfolio_catalog=private_portfolio_catalog,
    )
    specialist_context = _format_specialist_context(public_repo_catalog=public_repo_catalog)

    graph = create_portfolio_graph(
        search_tool=search_tool,
        specialist_tools=specialist_tools,
    )

    app.state.rag_service = rag_service
    app.state.graph = graph
    app.state.assistant_context = assistant_context
    app.state.specialist_context = specialist_context

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
    allow_origins=["*"],
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
