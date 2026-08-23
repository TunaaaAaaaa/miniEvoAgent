import sys

from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evo_repro import (
    BaseLLM,
    LLMResponse,
    PromptFeedbackExample,
    PromptRewriter,
)


class DemoOptimizerLLM(BaseLLM):
    def generate(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content=(
                "Answer factual questions with a concise final answer. "
                "Use the task wording and avoid extra explanation unless requested."
            ),
            metadata={"provider": "demo-fake"},
        )


def main() -> None:
    current_prompt = "You are a helpful assistant."
    examples = [
        PromptFeedbackExample(
            input="What is the capital of France?",
            prediction="France is a country in Europe.",
            label="Paris",
            metrics={"f1": 0.0, "em": 0.0, "acc": 0.0},
        ),
        PromptFeedbackExample(
            input="Who wrote Hamlet?",
            prediction="It was written by an English playwright.",
            label="William Shakespeare",
            metrics={"f1": 0.4, "em": 0.0, "acc": 0.0},
        ),
    ]

    rewriter = PromptRewriter(optimizer_llm=DemoOptimizerLLM())
    result = rewriter.rewrite(current_prompt=current_prompt, examples=examples)

    print("===== CURRENT PROMPT =====")
    print(result.original_prompt)
    print()
    print("===== FEEDBACK EXAMPLES =====")
    for index, example in enumerate(examples, start=1):
        print(f"Example {index}")
        print(f"Input: {example.input}")
        print(f"Prediction: {example.prediction}")
        print(f"Ground Truth: {example.label}")
        if example.metrics is not None:
            print(f"Metrics: {example.metrics}")
        print()
    print("===== CANDIDATE PROMPT =====")
    print(result.candidate_prompt)


if __name__ == "__main__":
    main()
