import copy
from typing import Any

import pytest

from evo_repro import (
    Action,
    BaseLLM,
    LLMResponse,
    PromptTemplate,
    TextOutputParser,
    ToolCall,
    get_word_length_tool,
)


class FakeLLM(BaseLLM):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "prompt": prompt,
                "messages": copy.deepcopy(messages),
                "tools": tools,
            }
        )
        return self.responses.pop(0)


def build_tool_action(llm: BaseLLM, max_tool_rounds: int = 3) -> Action:
    return Action(
        name="answer_with_tools",
        prompt_template=PromptTemplate(template="Question: {question}"),
        llm=llm,
        output_parser=TextOutputParser(),
        tools=[get_word_length_tool],
        max_tool_rounds=max_tool_rounds,
    )


def test_action_runs_llm_tool_llm_loop() -> None:
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_word_length",
                        arguments={"text": "retrieval"},
                    )
                ],
            ),
            LLMResponse(content="The word retrieval has 9 characters."),
        ]
    )
    action = build_tool_action(llm)

    output = action.execute(
        {"question": 'How many characters are in the word "retrieval"?'}
    )

    assert output.content == "The word retrieval has 9 characters."
    assert len(llm.calls) == 2
    assert llm.calls[0]["messages"] == [
        {
            "role": "user",
            "content": 'Question: How many characters are in the word "retrieval"?',
        }
    ]
    second_messages = llm.calls[1]["messages"]
    assert second_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "get_word_length",
        "content": "9",
    }
    assert output.metadata["tool_rounds"] == 1
    assert output.metadata["tool_results"][0]["result"] == 9


def test_action_keeps_phase_1_path_when_no_tool_call() -> None:
    llm = FakeLLM([LLMResponse(content="Plain answer.")])
    action = build_tool_action(llm)

    output = action.execute({"question": "No tool needed"})

    assert output.content == "Plain answer."
    assert len(llm.calls) == 1
    assert output.metadata["tool_rounds"] == 0
    assert output.metadata["tool_results"] == []


def test_action_unknown_tool_raises_clear_error() -> None:
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="unknown_tool",
                        arguments={},
                    )
                ],
            )
        ]
    )
    action = build_tool_action(llm)

    with pytest.raises(ValueError, match="Tool not found: unknown_tool"):
        action.execute({"question": "Use unknown tool"})


def test_action_max_tool_rounds_stops_infinite_calls() -> None:
    llm = FakeLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_word_length",
                        arguments={"text": "retrieval"},
                    )
                ],
            ),
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_2",
                        name="get_word_length",
                        arguments={"text": "retrieval"},
                    )
                ],
            ),
        ]
    )
    action = build_tool_action(llm, max_tool_rounds=1)

    with pytest.raises(RuntimeError, match="Exceeded max_tool_rounds=1"):
        action.execute({"question": "Keep calling tools"})
