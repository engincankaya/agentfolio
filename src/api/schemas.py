from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    session_id: str = Field(default="default", description="Conversation session ID")


class Source(BaseModel):
    file: str
    category: str
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    agent: str = ""


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    chunks_created: int
