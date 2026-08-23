import pytest

from typing import Any

from evo_repro import (
    BaseLLM,
    LLMResponse,
    PromptFeedbackExample,
    PromptRewriter,
)


class FakeOptimizerLLM(BaseLLM):
    def __init__(self, content: str) -> None:
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
        return LLMResponse(content=self.content)


def test_feedback_example_stores_expected_fields() -> None:
    example = PromptFeedbackExample(
        input="What is the capital of France?",
        prediction="France is in Europe.",
        label="Paris",
        metrics={"f1": 0.0, "em": 0.0, "acc": 0.0},
    )

    assert example.input == "What is the capital of France?"
    assert example.prediction == "France is in Europe."
    assert example.label == "Paris"
    assert example.metrics == {"f1": 0.0, "em": 0.0, "acc": 0.0}


def test_rewrite_calls_optimizer_llm_and_returns_candidate_prompt() -> None:
    optimizer_llm = FakeOptimizerLLM(
        "Answer questions concisely and provide only the final answer."
    )
    rewriter = PromptRewriter(optimizer_llm=optimizer_llm)

    result = rewriter.rewrite(
        current_prompt="You are a helpful assistant.",
        examples=[
            PromptFeedbackExample(
                input="Capital of France?",
                prediction="The capital is Paris.",
                label="Paris",
            )
        ],
    )

    assert optimizer_llm.call_count == 1
    assert result.original_prompt == "You are a helpful assistant."
    assert result.candidate_prompt == (
        "Answer questions concisely and provide only the final answer."
    )


def test_current_prompt_is_included_in_optimizer_prompt() -> None:
    optimizer_llm = FakeOptimizerLLM("Candidate prompt")
    rewriter = PromptRewriter(optimizer_llm=optimizer_llm)

    rewriter.rewrite(
        current_prompt="CURRENT SYSTEM PROMPT",
        examples=[
            PromptFeedbackExample(
                input="Question",
                prediction="Prediction",
                label="Label",
            )
        ],
    )

    assert optimizer_llm.last_prompt is not None
    assert "CURRENT SYSTEM PROMPT" in optimizer_llm.last_prompt


def test_examples_are_formatted_into_optimizer_prompt() -> None:
    optimizer_llm = FakeOptimizerLLM("Candidate prompt")
    rewriter = PromptRewriter(optimizer_llm=optimizer_llm)

    rewriter.rewrite(
        current_prompt="You are concise.",
        examples=[
            PromptFeedbackExample(
                input="Who wrote Hamlet?",
                prediction="A playwright wrote it.",
                label="William Shakespeare",
                metrics={"f1": 0.33, "em": 0.0, "acc": 0.0},
            )
        ],
    )

    assert optimizer_llm.last_prompt is not None
    assert "Input:\nWho wrote Hamlet?" in optimizer_llm.last_prompt
    assert "Agent Prediction:\nA playwright wrote it." in optimizer_llm.last_prompt
    assert "Ground Truth:\nWilliam Shakespeare" in optimizer_llm.last_prompt
    assert "F1: 0.33" in optimizer_llm.last_prompt


def test_multiple_examples_are_included_in_optimizer_prompt() -> None:
    optimizer_llm = FakeOptimizerLLM("Candidate prompt")
    rewriter = PromptRewriter(optimizer_llm=optimizer_llm)

    rewriter.rewrite(
        current_prompt="You are helpful.",
        examples=[
            PromptFeedbackExample(
                input="Question one",
                prediction="Prediction one",
                label="Label one",
            ),
            PromptFeedbackExample(
                input="Question two",
                prediction="Prediction two",
                label="Label two",
            ),
        ],
    )

    assert optimizer_llm.last_prompt is not None
    assert "## Example 1" in optimizer_llm.last_prompt
    assert "Question one" in optimizer_llm.last_prompt
    assert "## Example 2" in optimizer_llm.last_prompt
    assert "Question two" in optimizer_llm.last_prompt


def test_empty_examples_raise_value_error() -> None:
    rewriter = PromptRewriter(optimizer_llm=FakeOptimizerLLM("Candidate prompt"))

    with pytest.raises(ValueError, match="at least one feedback example"):
        rewriter.rewrite("You are helpful.", [])


def test_empty_candidate_prompt_raises_value_error() -> None:
    rewriter = PromptRewriter(optimizer_llm=FakeOptimizerLLM("   "))

    with pytest.raises(ValueError, match="empty candidate prompt"):
        rewriter.rewrite(
            "You are helpful.",
            [
                PromptFeedbackExample(
                    input="Question",
                    prediction="Prediction",
                    label="Label",
                )
            ],
        )
