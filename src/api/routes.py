from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from src.api.schemas import ChatRequest, ChatResponse
from src.api.streaming import (
    AnswerStreamExtractor,
    ndjson_event,
    normalize_stream_content,
    normalize_suggestions,
)
from src.core.logging import logger

router = APIRouter()


def get_rag_service(request: Request):
    return request.app.state.rag_service


def get_graph(request: Request):
    return request.app.state.graph


def get_github_user_context(request: Request):
    return getattr(request.app.state, "github_user_context", None)


def get_assistant_context(request: Request):
    return getattr(request.app.state, "assistant_context", None)


def get_specialist_context(request: Request):
    return getattr(request.app.state, "specialist_context", None)


def build_chat_config(
    body: ChatRequest,
    github_user_context,
    assistant_context: str | None,
    specialist_context: str | None,
) -> dict:
    return {
        "configurable": {
            "thread_id": body.session_id,
            "github_user_context": github_user_context,
            "assistant_context": assistant_context,
            "specialist_context": specialist_context,
        }
    }


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    graph=Depends(get_graph),
    rag_service=Depends(get_rag_service),
    github_user_context=Depends(get_github_user_context),
    assistant_context=Depends(get_assistant_context),
    specialist_context=Depends(get_specialist_context),
):
    try:
        config = build_chat_config(
            body=body,
            github_user_context=github_user_context,
            assistant_context=assistant_context,
            specialist_context=specialist_context,
        )

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config=config,
        )

        last_message = result["messages"][-1]
        agent_name = result.get("current_node", "assistant") or "assistant"
        response_payload = result.get("response") or {}

        answer = response_payload.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            answer = last_message.content if isinstance(last_message.content, str) else ""

        suggestions = response_payload.get("suggestions", [])
        suggestions = normalize_suggestions(suggestions)

        return ChatResponse(
            answer=answer,
            suggestions=suggestions,
            agent=agent_name,
        )

    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest,
    graph=Depends(get_graph),
    github_user_context=Depends(get_github_user_context),
    assistant_context=Depends(get_assistant_context),
    specialist_context=Depends(get_specialist_context),
):
    config = build_chat_config(
        body=body,
        github_user_context=github_user_context,
        assistant_context=assistant_context,
        specialist_context=specialist_context,
    )

    async def event_generator():
        final_state = None
        emitted_answer = ""
        extractor = AnswerStreamExtractor()

        try:
            async for mode, chunk in graph.astream(
                {"messages": [HumanMessage(content=body.message)]},
                config=config,
                stream_mode=["messages", "values"],
            ):
                if mode == "messages" and not extractor.done:
                    message_chunk, _metadata = chunk
                    content = normalize_stream_content(getattr(message_chunk, "content", ""))
                    delta = extractor.feed(content)
                    if delta:
                        emitted_answer += delta
                        yield ndjson_event("answer_delta", content=delta)

                if mode == "values" and isinstance(chunk, dict) and "messages" in chunk:
                    final_state = chunk

            response_payload = (final_state or {}).get("response") or {}
            final_answer = response_payload.get("answer")
            if isinstance(final_answer, str) and final_answer and not emitted_answer:
                yield ndjson_event("answer_delta", content=final_answer)

            yield ndjson_event(
                "done",
                agent=(final_state or {}).get("current_node", "assistant") or "assistant",
                suggestions=normalize_suggestions(response_payload.get("suggestions", [])),
            )
        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            yield ndjson_event("error", message=str(e))

    return StreamingResponse(
        event_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )
