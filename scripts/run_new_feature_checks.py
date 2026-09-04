"""Run smoke checks for the BaseModel/config feature set."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evo_repro import (
    Action,
    Agent,
    BaseLLM,
    EvolutionExample,
    LLMResponse,
    PromptRewriter,
    PromptTemplate,
    TextOutputParser,
    Tool,
)


class SmokeLLM(BaseLLM):
    def generate(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="ok", metadata={"provider": "smoke"})


def echo(value: str) -> str:
    return value


def assert_jsonable(config: dict[str, Any]) -> None:
    json.dumps(config)


def run_smoke_checks() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        example = EvolutionExample(
            sample_id="smoke_001",
            input="Capital of France?",
            label="Paris",
        )
        path = example.save_json(Path(temp_dir) / "example.json")
        assert EvolutionExample.from_json_file(path) == example

    tool = Tool(
        name="echo",
        description="Return the provided value.",
        parameters_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        function=echo,
    )
    tool_config = tool.to_config()
    assert "function" not in tool_config
    assert tool_config["function_ref"]["qualname"] == "echo"
    assert_jsonable(tool_config)

    action = Action(
        name="answer",
        prompt_template=PromptTemplate(template="Question: {question}"),
        llm=SmokeLLM(),
        output_parser=TextOutputParser(),
        tools=[tool],
    )
    action_config = action.to_config()
    assert "llm" not in action_config
    assert action_config["llm_config"] == {"class_name": "SmokeLLM"}
    assert_jsonable(action_config)

    agent = Agent(name="qa_agent", description="QA smoke test.", action=action)
    agent_config = agent.to_config()
    assert agent_config["action"]["tools"][0]["function_ref"]["qualname"] == "echo"
    assert_jsonable(agent_config)

    rewriter = PromptRewriter(optimizer_llm=SmokeLLM())
    rewriter_config = rewriter.to_config()
    assert "optimizer_llm" not in rewriter_config
    assert rewriter_config["optimizer_llm_config"] == {"class_name": "SmokeLLM"}
    assert_jsonable(rewriter_config)


def run_pytest() -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_base_model.py"],
        check=True,
    )


def main() -> None:
    run_smoke_checks()
    run_pytest()
    print("BaseModel/config smoke checks passed.")


if __name__ == "__main__":
    main()
