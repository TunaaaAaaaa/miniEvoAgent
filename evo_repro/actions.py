import json
from typing import Any

from pydantic import Field

from .base import BaseModel
from .llms import BaseLLM
from .parsers import ActionOutput, TextOutputParser
from .prompts import PromptTemplate
from .tools import Tool, ToolResult


class Action(BaseModel):
    """One executable task: render prompt, call LLM, parse output."""

    name: str
    prompt_template: PromptTemplate
    llm: BaseLLM
    output_parser: TextOutputParser
    tools: list[Tool] = Field(default_factory=list)
    max_tool_rounds: int = 3

    model_config = {"arbitrary_types_allowed": True}

    def execute(self, inputs: dict[str, object]) -> ActionOutput:
        prompt = self.prompt_template.format(**inputs)
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        llm_response = self.llm.generate(prompt, messages=messages, tools=self.tools)
        tool_results: list[ToolResult] = []
        tool_rounds = 0

        while llm_response.tool_calls:
            if tool_rounds >= self.max_tool_rounds:
                raise RuntimeError(
                    f"Exceeded max_tool_rounds={self.max_tool_rounds} "
                    f"for action '{self.name}'."
                )
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": llm_response.content or None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                    for tool_call in llm_response.tool_calls
                ],
            }
            messages.append(assistant_message)

            tool_map = {tool.name: tool for tool in self.tools}
            for tool_call in llm_response.tool_calls:
                tool = tool_map.get(tool_call.name)
                if tool is None:
                    raise ValueError(f"Tool not found: {tool_call.name}")
                tool_result = tool.execute(
                    tool_call.arguments,
                    tool_call_id=tool_call.id,
                )
                tool_results.append(tool_result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": json.dumps(tool_result.result),
                    }
                )

            tool_rounds += 1
            llm_response = self.llm.generate(prompt, messages=messages, tools=self.tools)

        output = self.output_parser.parse(llm_response)
        output.metadata.update(
            {
                "action": self.name,
                "prompt": prompt,
                "prompt_template": self.prompt_template.template,
                "input_keys": sorted(inputs),
                "tool_rounds": tool_rounds,
                "tool_results": [result.model_dump() for result in tool_results],
            }
        )
        return output

    def to_config(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "version": self.version,
            "name": self.name,
            "prompt_template": self.prompt_template.to_config(),
            "llm_config": self.llm.to_config(),
            "output_parser": self.output_parser.to_config(),
            "tools": [tool.to_config() for tool in self.tools],
            "max_tool_rounds": self.max_tool_rounds,
        }
