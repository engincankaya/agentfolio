from abc import ABC, abstractmethod

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.agents.output_schema import ChatTurnOutput
from src.agents.state import GraphState


class BaseAgentNode(ABC):
    """Abstract base class for all agent nodes in the handoff workflow."""

    def __init__(self, llm: ChatOpenAI, tools: list) -> None:
        self._llm = llm
        self._tools = tools
        self._structured_llm = self._llm.with_structured_output(
            ChatTurnOutput,
            method="json_schema",
            strict=True,
            include_raw=True,
            tools=self._tools,
        )

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System-level instruction that defines the agent's role and behaviour."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Node identifier used for current_node tracking."""

    def build_system_prompt(self, runtime_context: str | None = None) -> str:
        """Builds the final system prompt including runtime context when available."""
        if runtime_context and runtime_context.strip():
            return f"{self.system_prompt}\n\n<runtime_injection>\n{runtime_context.strip()}\n</runtime_injection>"
        return self.system_prompt

    async def _invoke_structured(self, messages: list) -> dict:
        result = await self._structured_llm.ainvoke(messages)
        raw_message = result["raw"]

        if getattr(raw_message, "tool_calls", None):
            return {
                "messages": [raw_message],
                "current_node": self.name,
                "response": None,
            }

        parsed = result["parsed"]
        if parsed is None:
            answer = raw_message.content if isinstance(raw_message.content, str) else ""
            parsed = ChatTurnOutput(answer=answer, suggestions=[])

        suggestions = [item.strip() for item in parsed.suggestions if item and item.strip()][:3]
        return {
            "messages": [AIMessage(content=parsed.answer)],
            "current_node": self.name,
            "response": {
                "answer": parsed.answer,
                "suggestions": suggestions,
            },
        }

    async def process_message(self, state: GraphState) -> dict:
        """Runs the agent and returns messages + current_node update."""
        messages = [SystemMessage(content=self.build_system_prompt()), *state["messages"]]
        return await self._invoke_structured(messages)
