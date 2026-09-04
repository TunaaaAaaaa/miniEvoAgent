import json
import os
from abc import ABC, abstractmethod
from typing import Any

from pydantic import Field

from .base import BaseModel
from .urls import redact_url_secrets


class ToolCall(BaseModel):
    """Tool invocation requested by an LLM."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Raw response returned by an LLM provider."""

    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    raw_response: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseLLM(ABC):
    """Minimal synchronous LLM interface."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        """Generate a model response for a prompt."""

    def to_config(self) -> dict[str, Any]:
        """Return non-secret configuration metadata for this runtime LLM."""

        return {"class_name": type(self).__name__}


class OpenAILLM(BaseLLM):
    """Tiny OpenAI-compatible chat-completions adapter."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._load_env()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required.")
        self.api_key = resolved_api_key
        self.base_url = (
            base_url
            or api_base
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
        )

    def to_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "class_name": type(self).__name__,
            "model": self.model,
        }
        if self.base_url:
            config["base_url"] = redact_url_secrets(self.base_url)
        return config

    def _load_env(self) -> None:
        try:
            from dotenv import load_dotenv
        except ImportError:
            return
        load_dotenv()

    def generate(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the openai package to use OpenAILLM.") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        request_messages = messages or [{"role": "user", "content": prompt}]
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
        }
        if tools:
            request_kwargs["tools"] = [tool.to_schema() for tool in tools]
        response = client.chat.completions.create(
            **request_kwargs,
        )
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"OpenAI returned invalid tool arguments for "
                    f"'{call.function.name}': {call.function.arguments}"
                ) from exc
            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw_response=response,
            metadata={"model": self.model},
        )
