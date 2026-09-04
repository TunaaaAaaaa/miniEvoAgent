from typing import Any

from pydantic import Field

from evo_repro.base import BaseModel
from evo_repro.llms import BaseLLM


class PromptFeedbackExample(BaseModel):
    """A single training feedback item for direct prompt rewriting."""

    sample_id: str | None = None
    input: str
    prediction: str
    label: str
    metrics: dict[str, float] | None = None


class PromptRewriteResult(BaseModel):
    """The original prompt and one rewritten candidate."""

    original_prompt: str
    candidate_prompt: str


class PromptRewriter(BaseModel):
    """Rewrite a system prompt from examples of agent performance."""

    optimizer_llm: BaseLLM = Field(description="LLM responsible for improving prompts.")

    model_config = {"arbitrary_types_allowed": True}

    def rewrite(
        self,
        current_prompt: str,
        examples: list[PromptFeedbackExample],
    ) -> PromptRewriteResult:
        if not examples:
            raise ValueError("Prompt rewriting requires at least one feedback example.")

        optimizer_prompt = self._build_optimizer_prompt(current_prompt, examples)
        response = self.optimizer_llm.generate(optimizer_prompt)
        candidate_prompt = response.content.strip()

        if not candidate_prompt:
            raise ValueError("Optimizer LLM returned an empty candidate prompt.")

        return PromptRewriteResult(
            original_prompt=current_prompt,
            candidate_prompt=candidate_prompt,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "version": self.version,
            "optimizer_llm_config": self.optimizer_llm.to_config(),
        }

    def _build_optimizer_prompt(
        self,
        current_prompt: str,
        examples: list[PromptFeedbackExample],
    ) -> str:
        formatted_examples = "\n\n".join(
            self._format_example(index, example)
            for index, example in enumerate(examples, start=1)
        )

        return (
            "You are improving the system prompt of another AI agent.\n\n"
            "Current system prompt:\n"
            f"{current_prompt}\n\n"
            "Below are examples of the agent's performance.\n\n"
            f"{formatted_examples}\n\n"
            "Rewrite the system prompt so the agent performs better on future tasks "
            "of the same type.\n\n"
            "Requirements:\n"
            "1. Improve general behavior rather than memorizing individual examples.\n"
            "2. Do not include the ground-truth answers from the examples in the prompt.\n"
            "3. Keep the prompt concise and reusable.\n"
            "4. Do not modify the agent architecture.\n"
            "5. Do not add or remove tools.\n"
            "6. Do not solve the examples directly.\n"
            "7. Return only the revised system prompt."
        )

    def _format_example(self, index: int, example: PromptFeedbackExample) -> str:
        parts = [
            f"## Example {index}",
            "",
            "Input:",
            example.input,
            "",
            "Agent Prediction:",
            example.prediction,
            "",
            "Ground Truth:",
            example.label,
        ]
        if example.metrics is not None:
            metric_lines = [
                f"{name.upper()}: {value}"
                for name, value in example.metrics.items()
            ]
            parts.extend(["", "Metrics:", *metric_lines])
        return "\n".join(parts)
