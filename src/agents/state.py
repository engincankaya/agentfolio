from typing import Annotated, List, Optional, TypedDict
import operator

from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """
    LangGraph State for the assistant + specialist workflow.
    - messages: Accumulated conversation history
    - current_node: Tracks which node is active (used by tool_router for return routing)
    """
    messages: Annotated[List[BaseMessage], operator.add]
    current_node: Optional[str]
