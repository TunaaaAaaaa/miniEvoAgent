import pytest

from types import SimpleNamespace
from typing import Any

from evo_repro import (
    Action,
    Agent,
    BaseLLM,
    LLMResponse,
    PromptTemplate,
    TextOutputParser,
)


class FakeLLM(BaseLLM):
    def __init__(self, content: str = "Paris") -> None:
        self.content = content
        self.last_prompt: str | None = None
        self.call_count = 0

    def generate(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        self.last_prompt = prompt
        self.call_count += 1
        return LLMResponse(content=self.content, metadata={"provider": "fake"})


def test_prompt_template_formats_values() -> None:
    template = PromptTemplate(template="Answer:\n{question}")

    assert template.format(question="What is RAG?") == "Answer:\nWhat is RAG?"


def test_prompt_template_missing_variable_has_clear_error() -> None:
    template = PromptTemplate(template="Answer:\n{question}")

    with pytest.raises(ValueError, match="Missing prompt variable\\(s\\): question"):
        template.format()


def test_prompt_template_supports_attribute_and_item_access() -> None:
    template = PromptTemplate(template="{user.name}: {values[0]}")

    assert template.required_variables() == {"user", "values"}
    assert template.format(
        user=SimpleNamespace(name="Ada"),
        values=["ready"],
    ) == "Ada: ready"


def test_prompt_template_discovers_variables_in_nested_format_specs() -> None:
    template = PromptTemplate(template="{value:.{precision}f}")

    assert template.required_variables() == {"value", "precision"}
    assert template.format(value=3.14159, precision=2) == "3.14"


def test_prompt_template_rejects_positional_variables() -> None:
    template = PromptTemplate(template="Answer: {}")

    with pytest.raises(ValueError, match="supports named variables only"):
        template.format()


def test_action_executes_prompt_llm_and_parser() -> None:
    llm = FakeLLM()
    action = Action(
        name="answer_question",
        prompt_template=PromptTemplate(template="Question: {question}"),
        llm=llm,
        output_parser=TextOutputParser(),
    )

    output = action.execute({"question": "Capital of France?"})

    assert llm.last_prompt == "Question: Capital of France?"
    assert output.content == "Paris"
    assert output.raw_response is None
    assert output.metadata["prompt"] == "Question: Capital of France?"
    assert output.metadata["llm"] == {"provider": "fake"}
    assert output.metadata["input_keys"] == ["question"]


def test_agent_executes_action_and_returns_message() -> None:
    llm = FakeLLM()
    action = Action(
        name="answer_question",
        prompt_template=PromptTemplate(template="Question: {question}"),
        llm=llm,
        output_parser=TextOutputParser(),
    )
    agent = Agent(
        name="qa_agent",
        description="Answers questions.",
        action=action,
    )

    message = agent.execute({"question": "Capital of France?"})

    assert message.agent == "qa_agent"
    assert message.action == "answer_question"
    assert message.content == "Paris"
    assert message.metadata["prompt"] == "Question: Capital of France?"
    assert message.metadata["llm"] == {"provider": "fake"}
    assert llm.call_count == 1


def test_message_records_expected_fields() -> None:
    action = Action(
        name="answer_question",
        prompt_template=PromptTemplate(template="{question}"),
        llm=FakeLLM("A short answer"),
        output_parser=TextOutputParser(),
    )
    agent = Agent(name="qa_agent", description="QA", action=action)

    message = agent.execute({"question": "Say something"})

    assert message.agent == "qa_agent"
    assert message.action == "answer_question"
    assert message.content == "A short answer"
    assert message.metadata["description"] == "QA"
