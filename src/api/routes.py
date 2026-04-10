from fastapi import APIRouter, HTTPException, Request, Depends
from langchain_core.messages import HumanMessage

from src.api.schemas import ChatRequest, ChatResponse, Source
from src.core.logging import logger

router = APIRouter()


def get_rag_service(request: Request):
    return request.app.state.rag_service


def get_graph(request: Request):
    return request.app.state.graph


def get_github_user_context(request: Request):
    return getattr(request.app.state, "github_user_context", None)


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    graph=Depends(get_graph),
    rag_service=Depends(get_rag_service),
    github_user_context=Depends(get_github_user_context),
):
    try:
        config = {
            "configurable": {
                "thread_id": body.session_id,
                "github_user_context": github_user_context,
            }
        }

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config=config,
        )

        last_message = result["messages"][-1]
        agent_name = result.get("current_node", "chat_node") or "chat_node"

        raw_results = rag_service.similarity_search(body.message)
        sources = [
            Source(
                file=doc.metadata.get("filename", ""),
                category=doc.metadata.get("category", ""),
                snippet=doc.page_content[:200],
            )
            for doc, _ in raw_results
        ]

        return ChatResponse(answer=last_message.content, sources=sources, agent=agent_name)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


