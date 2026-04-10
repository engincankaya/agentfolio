from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from src.agents.state import GraphState
from src.agents.chat_node import ChatNode
from src.agents.github_agent import GitHubAgent
from src.agents.calendar_agent import CalendarAgent
from src.tools.portfolio_tools import handoff_to_github, handoff_to_calendar
from src.core.config import settings
from src.core.logging import logger


# ---------------------------------------------------------------------------
# Router functions
# ---------------------------------------------------------------------------

def chat_router(state: GraphState) -> str:
    """
    Routes ChatNode output.
    - If tool call exists → "tools"
    - Otherwise → END (direct response)
    """
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


def tool_router(state: GraphState) -> str:
    """
    Routes tools node output based on last executed tool.
    - handoff_to_github → "github_agent"
    - handoff_to_calendar → "calendar_agent"
    - search_portfolio → back to calling node (current_node)
    """
    messages = state.get("messages", [])

    recent_tool_names = []
    for msg in reversed(messages):
        if msg.type == "tool":
            recent_tool_names.append(msg.name)
        else:
            break

    print('recent_tool_names',recent_tool_names)

    if "handoff_to_github" in recent_tool_names:
        return "github_agent"

    if "handoff_to_calendar" in recent_tool_names:
        return "calendar_agent"

    current = state.get("current_node", "chat_node")
    return current


def agent_router(state: GraphState) -> str:
    """
    Routes GitHubAgent/CalendarAgent output.
    - If tool call exists → "tools"
    - Otherwise → END (answer is ready)
    """
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# ---------------------------------------------------------------------------
# Node wrappers (set current_node in state)
# ---------------------------------------------------------------------------

def _create_chat_node_fn(chat_instance: ChatNode):
    def chat_node_fn(state: GraphState) -> dict:
        return chat_instance.process_message(state)
    return chat_node_fn


def _create_github_agent_fn(github_instance: GitHubAgent):
    async def github_agent_fn(state: GraphState, config: RunnableConfig) -> dict:
        return await github_instance.process_message(state, config)
    return github_agent_fn


def _create_calendar_agent_fn(calendar_instance: CalendarAgent):
    async def calendar_agent_fn(state: GraphState) -> dict:
        return await calendar_instance.process_message(state)
    return calendar_agent_fn


def _create_tools_node(all_tools: list):
    tool_node = ToolNode(all_tools)

    async def tools_node_fn(state: GraphState) -> dict:
        result = await tool_node.ainvoke(state)
        return {"messages": result["messages"]}
    return tools_node_fn


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def create_portfolio_graph(
    search_tool,
    github_tools: list | None = None,
    calendar_tools: list | None = None,
    llm: ChatOpenAI | None = None,
    checkpointer=None,
) -> CompiledStateGraph:
    """
    Builds and compiles the handoff-based multi-agent workflow.

    Graph flow:
        START → chat_node → (chat_router) → tools / END
        tools → (tool_router) → github_agent / calendar_agent / chat_node
        github_agent/calendar_agent → (agent_router) → tools / END
    """
    if llm is None:
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
        )

    if checkpointer is None:
        checkpointer = MemorySaver()

    github_tools = github_tools or []
    calendar_tools = calendar_tools or []

    # Chat node tools: search + handoffs
    chat_tools = [search_tool, handoff_to_github, handoff_to_calendar]

    # All tools (for the shared ToolNode)
    all_tools = chat_tools + github_tools + calendar_tools

    # Agent instances
    chat = ChatNode(llm=llm, tools=chat_tools)
    github = GitHubAgent(llm=llm, tools=github_tools)
    calendar = CalendarAgent(llm=llm, tools=calendar_tools)

    # Build graph
    workflow = StateGraph(GraphState)

    workflow.add_node("chat_node", _create_chat_node_fn(chat))
    workflow.add_node("tools", _create_tools_node(all_tools))
    workflow.add_node("github_agent", _create_github_agent_fn(github))
    workflow.add_node("calendar_agent", _create_calendar_agent_fn(calendar))

    # Entry point
    workflow.set_entry_point("chat_node")

    # Chat Node → tools or END
    workflow.add_conditional_edges(
        "chat_node",
        chat_router,
        {"tools": "tools", END: END},
    )

    # Tools → which agent to go to?
    workflow.add_conditional_edges(
        "tools",
        tool_router,
        {
            "github_agent": "github_agent",
            "calendar_agent": "calendar_agent",
            "chat_node": "chat_node",
        },
    )

    # GitHub/Calendar Agent → tools or END
    workflow.add_conditional_edges(
        "github_agent",
        agent_router,
        {"tools": "tools", END: END},
    )
    workflow.add_conditional_edges(
        "calendar_agent",
        agent_router,
        {"tools": "tools", END: END},
    )

    graph = workflow.compile(checkpointer=checkpointer)
    logger.info("Agenticfolio graph compiled successfully.")
    return graph
