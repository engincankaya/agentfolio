# Agenticfolio

An open-source, AI-powered portfolio chatbot that lets anyone turn their professional background into an interactive conversational experience. Powered by a multi-agent architecture, it answers questions about your career, projects, and GitHub activity using RAG and live GitHub data.

Built with LangGraph, LangChain, and FastAPI.

## How It Works

Agenticfolio uses a multi-agent system where each agent specializes in a different domain:

```
User Question
     |
 Chat Node (router)
     |
     +-- Portfolio questions --> RAG search over your .md files
     +-- GitHub questions ----> GitHub Agent (live data via MCP)
     +-- Calendar questions --> Calendar Agent (Google Calendar via MCP)
```

- **Chat Node** greets the user, understands intent, and routes to the right specialist.
- **GitHub Agent** queries your public repositories, commits, README files, and code using the GitHub MCP server.
- **Calendar Agent** checks your Google Calendar for events and availability.
- **RAG pipeline** ingests your markdown files into a vector database for semantic search.

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for GitHub MCP server)
- A GitHub Personal Access Token
- An OpenRouter API key (or any OpenAI-compatible provider)

### 1. Clone and install

```bash
git clone https://github.com/your-username/agenticfolio.git
cd agenticfolio
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your API keys (see [Configuration](#configuration) below).

### 3. Add your portfolio content

Create markdown files in `data/raw/` describing your professional background (see [Writing Your Portfolio](#writing-your-portfolio) below).

### 4. Ingest your data

```bash
python scripts/ingest.py
```

This processes all markdown files in `data/raw/`, splits them into chunks, and stores them in the local Qdrant vector database.

### 5. Start the server and chat

```bash
python -m src.main
```

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What projects has this person worked on?", "session_id": "test-1"}'
```

## Writing Your Portfolio

Your portfolio lives in `data/raw/` as markdown files. The RAG pipeline ingests these files, splits them into chunks, and stores them in a vector database for semantic search.

### File structure

```
data/
  raw/
    about_me.md           # Bio, skills, career summary
    my_project.md         # One file per project
    another_project.md
    ...
```

### Guidelines

**about_me.md** - Your professional profile. Include:

```markdown
# About Me

## Professional Summary
A brief overview of who you are, your role, and your career goals.

## Skills
- Languages: Python, TypeScript, Go
- Frameworks: FastAPI, React, LangChain
- Databases: PostgreSQL, Redis, Qdrant

## Experience
### Company Name - Role (Start - End)
What you did, what you built, key achievements.

## Education
### University Name - Degree (Year)
```

**Project files** (`project_name.md`) - One file per project. Include:

```markdown
# Project Overview
What the project does and why it exists.

## Technology Stack
### Core Languages & Frameworks
- **Python 3.11+** - Primary language
- **FastAPI** - REST API framework

### Database & Infrastructure
- **PostgreSQL** - Primary database
- **Docker** - Containerized deployment

## Key Features & Capabilities
- Built a real-time notification system using WebSockets
- Implemented role-based access control with JWT authentication
- Designed a modular plugin architecture for third-party integrations
```

### Tips

- **Be specific.** Write concrete facts: technologies, metrics, architecture decisions. The chatbot can only answer based on what you write.
- **One project per file.** This helps the RAG pipeline find relevant chunks accurately.
- **Use frontmatter metadata.** Put project, company, role, period, visibility, and summary fields at the top of each markdown file.
- **Use headings.** Structured markdown with `##` headings helps chunking and retrieval quality.
- **Skip the fluff.** Focus on what you built and how, not generic adjectives.

## Configuration

All configuration is done through environment variables. Copy `.env.example` to `.env` and fill in the values.

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | API key for LLM provider (OpenRouter) |
| `LLM_MODEL` | No | LLM model to use (default: `google/gemini-3.1-flash-lite-preview`) |
| `EMBEDDING_MODEL` | No | Embedding model (default: `openai/text-embedding-3-small`) |
| `GITHUB_PAT` | Yes | GitHub Personal Access Token for MCP server |
| `MINDMAP_MCP_PACKAGE` | No | NPM package name for Mindmap MCP server, run with `npx -y <package>` |
| `MINDMAP_MCP_SERVER_PATH` | No | Local path to Mindmap MCP server `dist/index.js`, used only when `MINDMAP_MCP_PACKAGE` is empty |
| `GOOGLE_OAUTH_CLIENT_ID` | No | Google OAuth client ID for Calendar agent |
| `GOOGLE_OAUTH_CLIENT_SECRET` | No | Google OAuth client secret for Calendar agent |
| `QDRANT_URL` | No | Qdrant Cloud cluster URL. When set, remote Qdrant is used instead of local storage. |
| `QDRANT_API_KEY` | No | Qdrant Cloud API key |
| `QDRANT_PATH` | No | Local embedded Qdrant storage path, used only when `QDRANT_URL` is empty |
| `QDRANT_COLLECTION` | No | Qdrant collection name (default: `portfolio`) |
| `LANGSMITH_API_KEY` | No | LangSmith API key for tracing |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing (`true`/`false`) |

## Architecture

```
src/
  main.py                 # FastAPI app, lifespan, MCP setup
  agents/
    state.py              # LangGraph state definition
    graph.py              # Multi-agent graph builder
    base_agent.py         # Abstract base for agents
    chat_node.py          # Entry point / router agent
    github_agent.py       # GitHub specialist (MCP tools)
    calendar_agent.py     # Google Calendar specialist (MCP tools)
  api/
    routes.py             # REST endpoints
    schemas.py            # Request/response models
  core/
    config.py             # Pydantic settings
    logging.py            # Logger setup
  services/
    rag_service.py        # Vector store / RAG pipeline
    embedding_service.py  # Embedding model (singleton)
  tools/
    portfolio_tools.py    # Search & handoff tools
data/
  raw/                    # Your portfolio markdown files
  vector_db/              # Qdrant local storage (auto-generated)
```

## Tech Stack

- **LangGraph** - Multi-agent orchestration
- **LangChain** - LLM framework
- **FastAPI** - Async REST API
- **Qdrant** - Vector database for RAG
- **GitHub MCP Server** - Live GitHub data access
- **OpenRouter** - LLM provider (swap with any OpenAI-compatible API)

## License

MIT
