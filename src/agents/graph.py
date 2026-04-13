from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from src.agents.state import GraphState
from src.agents.assistant import AssistantNode
from src.agents.specialist import SpecialistNode
from src.tools.portfolio_tools import handoff_to_specialist
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
    - handoff_to_specialist → "specialist"
    - search_portfolio → back to calling node (current_node)
    """
    messages = state.get("messages", [])

    recent_tool_names = []
    for msg in reversed(messages):
        if msg.type == "tool":
            recent_tool_names.append(msg.name)
        else:
            break

    if "handoff_to_specialist" in recent_tool_names:
        return "specialist"

    current = state.get("current_node", "assistant")
    return current


def agent_router(state: GraphState) -> str:
    """
    Routes Specialist output.
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

def _create_assistant_node_fn(assistant_instance: AssistantNode):
    def assistant_node_fn(state: GraphState, config: RunnableConfig | None = None) -> dict:
        assistant_context = None
        if config:
            assistant_context = config.get("configurable", {}).get("assistant_context")
        return assistant_instance.process_message(state, assistant_context=assistant_context)

    return assistant_node_fn


def _create_specialist_node_fn(specialist_instance: SpecialistNode):
    async def specialist_node_fn(state: GraphState, config: RunnableConfig) -> dict:
        return await specialist_instance.process_message(state, config=config)

    return specialist_node_fn


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
    specialist_tools: list | None = None,
    llm: ChatOpenAI | None = None,
    checkpointer=None,
) -> CompiledStateGraph:
    """
    Builds and compiles the assistant + specialist workflow.

    Graph flow:
        START → assistant → (chat_router) → tools / END
        tools → (tool_router) → specialist / assistant
        specialist → (agent_router) → tools / END
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

    specialist_tools = specialist_tools or []

    assistant_tools = [search_tool, handoff_to_specialist]
    all_tools = assistant_tools + specialist_tools

    assistant = AssistantNode(llm=llm, tools=assistant_tools)
    specialist = SpecialistNode(llm=llm, tools=specialist_tools)

    # Build graph
    workflow = StateGraph(GraphState)

    workflow.add_node("assistant", _create_assistant_node_fn(assistant))
    workflow.add_node("tools", _create_tools_node(all_tools))
    workflow.add_node("specialist", _create_specialist_node_fn(specialist))

    # Entry point
    workflow.set_entry_point("assistant")

    # Chat Node → tools or END
    workflow.add_conditional_edges(
        "assistant",
        chat_router,
        {"tools": "tools", END: END},
    )

    # Tools → which agent to go to?
    workflow.add_conditional_edges(
        "tools",
        tool_router,
        {
            "specialist": "specialist",
            "assistant": "assistant",
        },
    )

    # Specialist → tools or END
    workflow.add_conditional_edges(
        "specialist",
        agent_router,
        {"tools": "tools", END: END},
    )

    graph = workflow.compile(checkpointer=checkpointer)
    logger.info("Assistant + specialist graph compiled successfully.")
    return graph
