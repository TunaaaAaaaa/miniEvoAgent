import os
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Raw text returned by an LLM provider."""

    content: str
    raw_response: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseLLM(ABC):
    """Minimal synchronous LLM interface."""

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a model response for a prompt."""


class OpenAILLM(BaseLLM):
    """Tiny OpenAI chat-completions adapter using OPENAI_API_KEY."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required.")

    def generate(self, prompt: str) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the openai package to use OpenAILLM.") from exc

        client_kwargs: dict[str, str] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content or ""
        return LLMResponse(
            content=content,
            raw_response=response,
            metadata={"model": self.model},
        )
