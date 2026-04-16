from pydantic import BaseModel, Field


class ChatTurnOutput(BaseModel):
    """Structured final response returned by assistant and specialist nodes."""

    answer: str = Field(..., description="Direct answer shown to the user.")
    suggestions: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to 3 short follow-up suggestions for the current topic.",
    )
