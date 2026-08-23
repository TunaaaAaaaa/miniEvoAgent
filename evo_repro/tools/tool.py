from typing import Any, Callable

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result returned by a tool execution."""

    tool_call_id: str
    name: str
    result: Any
    metadata: dict[str, Any] = Field(default_factory=dict)


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
            result = self.function(**arguments)
        except TypeError as exc:
            raise ValueError(
                f"Invalid arguments for tool '{self.name}': {arguments}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Tool '{self.name}' execution failed: {exc}") from exc
        return ToolResult(
            tool_call_id=tool_call_id,
            name=self.name,
            result=result,
            metadata={"arguments": arguments},
        )

    def to_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
