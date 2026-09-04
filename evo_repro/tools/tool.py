import inspect
import json
from typing import Any, Callable

from pydantic import Field

from evo_repro.base import BaseModel


class ToolResult(BaseModel):
    """Result returned by a tool execution."""

    tool_call_id: str
    name: str
    result: Any
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_message_content(self) -> str:
        """Serialize a tool result for an OpenAI-compatible tool message."""

        try:
            return json.dumps(self.result, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Tool '{self.name}' returned a non-JSON-serializable result of type "
                f"{type(self.result).__name__}."
            ) from exc


class Tool(BaseModel):
    """Minimal wrapper around a Python function callable by an LLM."""

    name: str
    description: str
    parameters_schema: dict[str, Any]
    function: Callable[..., Any]

    model_config = {"arbitrary_types_allowed": True}

    def execute(
        self,
        arguments: dict[str, Any],
        tool_call_id: str = "",
    ) -> ToolResult:
        if not isinstance(arguments, dict):
            raise ValueError(
                f"Tool '{self.name}' arguments must be a dict, got "
                f"{type(arguments).__name__}."
            )
        try:
            function_signature = inspect.signature(self.function)
        except (TypeError, ValueError):
            function_signature = None
        if function_signature is not None:
            try:
                function_signature.bind(**arguments)
            except TypeError as exc:
                raise ValueError(
                    f"Invalid arguments for tool '{self.name}': {arguments}"
                ) from exc

        try:
            result = self.function(**arguments)
        except Exception as exc:
            raise RuntimeError(f"Tool '{self.name}' execution failed: {exc}") from exc
        tool_result = ToolResult(
            tool_call_id=tool_call_id,
            name=self.name,
            result=result,
            metadata={"arguments": arguments},
        )
        tool_result.to_message_content()
        return tool_result

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def to_config(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "parameters_schema": self.parameters_schema,
            "function_ref": {
                "module": self.function.__module__,
                "qualname": self.function.__qualname__,
            },
        }
