from collections.abc import Callable, Sequence
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from evo_repro.agents import Agent
from evo_repro.evaluators import evaluate_qa

from .prompt_rewriter import PromptFeedbackExample, PromptRewriter
from .selector import PromptSelector, SelectionResult


class EvolutionExample(BaseModel):
    """One labeled QA sample used by the evolution loop."""

    sample_id: str
    input: str
    label: str


class EvaluationRecord(BaseModel):
    """Per-sample execution and evaluator output."""

    sample_id: str
    input: str
    prediction: str
    label: str
    metrics: dict[str, float]
    rendered_prompt: str | None = None


class RewriteTrace(BaseModel):
    """The exact training feedback passed into the prompt rewriter."""

    current_prompt: str
    feedback_examples: list[PromptFeedbackExample]
    optimizer_meta_prompt: str | None = None
    rewriter_training_sample_ids: list[str] = Field(default_factory=list)


class EvolutionRoundTrace(BaseModel):
    """Debugging trace for one self-evolution generation."""

    generation: int
    parent_prompt: str
    train_records: list[EvaluationRecord]
    rewrite_input: RewriteTrace
    candidate_prompt: str
    parent_prompt_used: str
    candidate_prompt_used: str
    parent_dev_records: list[EvaluationRecord]
    parent_dev_score: float
    candidate_dev_records: list[EvaluationRecord]
    candidate_dev_score: float
    selection: SelectionResult
    dev_sample_ids: list[str]

    @property
    def selected_prompt(self) -> str:
        return self.selection.selected_prompt


def evolve_once(
    *,
    parent_prompt: str,
    train_examples: Sequence[EvolutionExample | dict[str, Any]],
    dev_examples: Sequence[EvolutionExample | dict[str, Any]],
    agent_factory: Callable[[str], Agent],
    rewriter: PromptRewriter,
    selector: PromptSelector | None = None,
    generation: int = 0,
) -> EvolutionRoundTrace:
    """Run one train-feedback, rewrite, dev-selection evolution generation."""

    normalized_train = _normalize_examples(train_examples)
    normalized_dev = _normalize_examples(dev_examples)
    selector = selector or PromptSelector()

    train_records = _run_examples(parent_prompt, normalized_train, agent_factory)
    feedback_examples = [
        PromptFeedbackExample(
            sample_id=record.sample_id,
            input=record.input,
            prediction=record.prediction,
            label=record.label,
            metrics=record.metrics,
        )
        for record in train_records
    ]

    rewrite_result = rewriter.rewrite(parent_prompt, feedback_examples)
    candidate_prompt = rewrite_result.candidate_prompt
    optimizer_meta_prompt = getattr(rewriter.optimizer_llm, "last_prompt", None)

    parent_dev_records = _run_examples(parent_prompt, normalized_dev, agent_factory)
    candidate_dev_records = _run_examples(candidate_prompt, normalized_dev, agent_factory)
    parent_dev_score = _mean_f1(parent_dev_records)
    candidate_dev_score = _mean_f1(candidate_dev_records)

    selection = selector.select(
        parent_prompt=parent_prompt,
        parent_score=parent_dev_score,
        candidate_prompt=candidate_prompt,
        candidate_score=candidate_dev_score,
    )

    dev_sample_ids = [example.sample_id for example in normalized_dev]
    return EvolutionRoundTrace(
        generation=generation,
        parent_prompt=parent_prompt,
        train_records=train_records,
        rewrite_input=RewriteTrace(
            current_prompt=parent_prompt,
            feedback_examples=feedback_examples,
            optimizer_meta_prompt=optimizer_meta_prompt,
            rewriter_training_sample_ids=[
                example.sample_id for example in feedback_examples if example.sample_id
            ],
        ),
        candidate_prompt=candidate_prompt,
        parent_prompt_used=parent_prompt,
        candidate_prompt_used=candidate_prompt,
        parent_dev_records=parent_dev_records,
        parent_dev_score=parent_dev_score,
        candidate_dev_records=candidate_dev_records,
        candidate_dev_score=candidate_dev_score,
        selection=selection,
        dev_sample_ids=dev_sample_ids,
    )


def evolve(
    *,
    initial_prompt: str,
    train_examples: Sequence[EvolutionExample | dict[str, Any]],
    dev_examples: Sequence[EvolutionExample | dict[str, Any]],
    agent_factory: Callable[[str], Agent],
    rewriter: PromptRewriter,
    generations: int = 3,
    selector: PromptSelector | None = None,
) -> tuple[str, list[EvolutionRoundTrace]]:
    """Run several generations, carrying each selected prompt forward."""

    current_prompt = initial_prompt
    history: list[EvolutionRoundTrace] = []
    selector = selector or PromptSelector()

    for generation in range(generations):
        trace = evolve_once(
            parent_prompt=current_prompt,
            train_examples=train_examples,
            dev_examples=dev_examples,
            agent_factory=agent_factory,
            rewriter=rewriter,
            selector=selector,
            generation=generation,
        )
        history.append(trace)
        current_prompt = trace.selected_prompt

    return current_prompt, history


def _normalize_examples(
    examples: Sequence[EvolutionExample | dict[str, Any]],
) -> list[EvolutionExample]:
    return [
        example if isinstance(example, EvolutionExample) else EvolutionExample(**example)
        for example in examples
    ]


def _run_examples(
    prompt: str,
    examples: Sequence[EvolutionExample],
    agent_factory: Callable[[str], Agent],
) -> list[EvaluationRecord]:
    agent = agent_factory(prompt)
    records: list[EvaluationRecord] = []
    for example in examples:
        message = agent.execute({"question": example.input})
        metrics = evaluate_qa(message.content, example.label)
        records.append(
            EvaluationRecord(
                sample_id=example.sample_id,
                input=example.input,
                prediction=message.content,
                label=example.label,
                metrics=metrics,
                rendered_prompt=message.metadata.get("prompt"),
            )
        )
    return records


def _mean_f1(records: Sequence[EvaluationRecord]) -> float:
    if not records:
        raise ValueError("Cannot score an empty evaluation set.")
    return mean(record.metrics["f1"] for record in records)
