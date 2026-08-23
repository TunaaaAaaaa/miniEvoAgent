from typing import Any

from evo_repro import (
    Action,
    Agent,
    BaseLLM,
    LLMResponse,
    PromptTemplate,
    TextOutputParser,
    ToolCall,
    get_word_length_tool,
)


class FakeLLM(BaseLLM):
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="get_word_length",
                        arguments={"text": "retrieval"},
                    )
                ],
            )
        return LLMResponse(content="The word retrieval has 9 characters.")


def test_agent_returns_final_message_after_tool_loop() -> None:
    llm = FakeLLM()
    action = Action(
        name="answer_with_tools",
        prompt_template=PromptTemplate(template="Question: {question}"),
        llm=llm,
        output_parser=TextOutputParser(),
        tools=[get_word_length_tool],
    )
    agent = Agent(
        name="tool_agent",
        description="Answers questions with tools.",
        action=action,
    )

    message = agent.execute(
        {"question": 'How many characters are in the word "retrieval"?'}
    )

    assert message.content == "The word retrieval has 9 characters."
    assert llm.calls == 2
    assert message.metadata["tool_rounds"] == 1
    assert message.metadata["tool_results"][0]["result"] == 9
