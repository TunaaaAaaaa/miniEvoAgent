import pytest

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

    def generate(self, prompt: str) -> LLMResponse:
        self.last_prompt = prompt
        return LLMResponse(content=self.content, metadata={"provider": "fake"})


def test_prompt_template_formats_values() -> None:
    template = PromptTemplate(template="Answer:\n{question}")

    assert template.format(question="What is RAG?") == "Answer:\nWhat is RAG?"


def test_prompt_template_missing_variable_has_clear_error() -> None:
    template = PromptTemplate(template="Answer:\n{question}")

    with pytest.raises(ValueError, match="Missing prompt variable\\(s\\): question"):
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


def test_agent_executes_action_and_returns_message() -> None:
    action = Action(
        name="answer_question",
        prompt_template=PromptTemplate(template="Question: {question}"),
        llm=FakeLLM(),
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
