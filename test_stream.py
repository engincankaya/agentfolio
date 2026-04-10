import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.services.embedding_service import EmbeddingService
from src.services.rag_service import RAGService
from src.tools.portfolio_tools import build_search_tool
from src.agents.graph import create_portfolio_graph
from langchain_core.messages import HumanMessage

rag_service = RAGService(cfg=settings, embedding_service=EmbeddingService(cfg=settings))
search_tool = build_search_tool(rag_service=rag_service)
graph = create_portfolio_graph(search_tool=search_tool)

for output in graph.stream(
    {"messages": [HumanMessage(content="Engincan kimdir?")]},
    config={"configurable": {"thread_id": "test_stream"}},
):
    print("STEP OUTPUT:", list(output.keys()))
    for key, value in output.items():
        print(f"Node: {key}")
        if "messages" in value:
            print("Messages:", [m.content[:100] for m in value["messages"]])
        if "current_node" in value:
            print("Current node:", value["current_node"])
        print("-------")
