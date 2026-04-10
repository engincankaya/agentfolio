from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from src.agents.state import GraphState


class BaseAgentNode(ABC):
    """Abstract base class for all agent nodes in the handoff workflow."""

    def __init__(self, llm: ChatOpenAI, tools: list) -> None:
        self._llm = llm
        self._tools = tools
        self._executor = create_react_agent(
            self._llm,
            self._tools,
            prompt=SystemMessage(content=self.system_prompt),
        )

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System-level instruction that defines the agent's role and behaviour."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Node identifier used for current_node tracking."""

    async def process_message(self, state: GraphState) -> dict:
        """Runs the agent and returns messages + current_node update."""
        result = await self._executor.ainvoke({"messages": state["messages"]})
        return {
            "messages": result["messages"],
            "current_node": self.name,
        }
