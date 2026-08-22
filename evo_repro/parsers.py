from typing import Any

from pydantic import BaseModel, Field

from .llms import LLMResponse


class ActionOutput(BaseModel):
    """Parsed action output."""

    content: str
    raw_response: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextOutputParser:
    """Parser that keeps the model text unchanged."""

    def parse(self, response: LLMResponse) -> ActionOutput:
        return ActionOutput(
            content=response.content,
            raw_response=response.raw_response,
            metadata={"llm": response.metadata},
        )
