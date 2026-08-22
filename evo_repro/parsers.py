from typing import Any

from pydantic import BaseModel

from .llms import LLMResponse


class ActionOutput(BaseModel):
    """Parsed action output."""

    content: str
    raw_response: Any | None = None


class TextOutputParser:
    """Parser that keeps the model text unchanged."""

    def parse(self, response: LLMResponse) -> ActionOutput:
        return ActionOutput(content=response.content, raw_response=response.raw_response)
