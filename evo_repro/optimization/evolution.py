from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

from pydantic import Field

from evo_repro.agents import Agent
from evo_repro.base import BaseModel
from evo_repro.provenance import RunProvenance
from evo_repro.storage import _json, stable_hash
from evo_repro.tasks import QATask, TaskAdapter

from .prompt_rewriter import PromptFeedbackExample, PromptRewriter
from .selector import PromptSelector, SelectionResult


class EvolutionExample(BaseModel):
    """One labeled sample used by the evolution loop."""

    sample_id: str
    input: Any
    label: Any


class EvaluationRecord(BaseModel):
    """Per-sample execution and evaluator output."""

    sample_id: str
    input: Any
    prediction: Any
    label: Any
    metrics: dict[str, float]
    rendered_prompt: str | None = None
    agent_inputs: dict[str, Any] = Field(default_factory=dict)
    agent_config: dict[str, Any] = Field(default_factory=dict)
    response_metadata: dict[str, Any] = Field(default_factory=dict)


class RewriteTrace(BaseModel):
    """The exact training feedback passed into the prompt rewriter."""

    current_prompt: str
    feedback_examples: list[PromptFeedbackExample]
    optimizer_meta_prompt: str | None = None
    rewriter_training_sample_ids: list[str] = Field(default_factory=list)
    optimizer_llm_config: dict[str, Any] = Field(default_factory=dict)
    response_metadata: dict[str, Any] = Field(default_factory=dict)
    response_content: str | None = None


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
    task_config: dict[str, Any] = Field(default_factory=dict)
    score_key: str = "f1"
    provenance: RunProvenance = Field(default_factory=RunProvenance)
    train_fingerprint: str | None = None
    dev_fingerprint: str | None = None

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
    task: TaskAdapter | None = None,
    provenance: RunProvenance | None = None,
) -> EvolutionRoundTrace:
    """Run one train-feedback, rewrite, dev-selection evolution generation."""

    normalized_train = _normalize_examples(train_examples)
    normalized_dev = _normalize_examples(dev_examples)
    if not normalized_train or not normalized_dev:
        raise ValueError("Training and development sets must be nonempty.")
    train_ids = {example.sample_id for example in normalized_train}
    dev_ids = {example.sample_id for example in normalized_dev}
    if (len(train_ids) != len(normalized_train) or len(dev_ids) != len(normalized_dev)
            or train_ids & dev_ids):
        raise ValueError("Sample IDs must be unique within and disjoint across train/dev.")
    selector = selector or PromptSelector()
    task = task or QATask()
    provenance = (provenance or RunProvenance.capture()).model_copy(deep=True)
    task_config = deepcopy(task.to_config())
    train_fingerprint = _fingerprint(normalized_train)
    dev_fingerprint = _fingerprint(normalized_dev)

    train_records = _run_examples(parent_prompt, normalized_train, agent_factory, task)
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
    parent_dev_records = _run_examples(parent_prompt, normalized_dev, agent_factory, task)
    candidate_dev_records = _run_examples(candidate_prompt, normalized_dev, agent_factory, task)
    parent_dev_score = task.aggregate([record.metrics for record in parent_dev_records])
    candidate_dev_score = task.aggregate([record.metrics for record in candidate_dev_records])

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
            optimizer_meta_prompt=rewrite_result.optimizer_prompt,
            optimizer_llm_config=rewrite_result.optimizer_llm_config,
            response_metadata=rewrite_result.response_metadata,
            response_content=rewrite_result.response_content,
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
        task_config=task_config,
        score_key=task.score_key,
        provenance=provenance,
        train_fingerprint=train_fingerprint,
        dev_fingerprint=dev_fingerprint,
        version=1,
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
    task: TaskAdapter | None = None,
    provenance: RunProvenance | None = None,
) -> tuple[str, list[EvolutionRoundTrace]]:
    """Run several generations, carrying each selected prompt forward."""

    current_prompt = initial_prompt
    history: list[EvolutionRoundTrace] = []
    selector = selector or PromptSelector()
    provenance = provenance or RunProvenance.capture()
    if generations < 0:
        raise ValueError("generations must be nonnegative.")

    for generation in range(generations):
        trace = evolve_once(
            parent_prompt=current_prompt,
            train_examples=train_examples,
            dev_examples=dev_examples,
            agent_factory=agent_factory,
            rewriter=rewriter,
            selector=selector,
            generation=generation,
            task=task,
            provenance=provenance,
        )
        history.append(trace)
        current_prompt = trace.selected_prompt

    return current_prompt, history


def _normalize_examples(
    examples: Sequence[EvolutionExample | dict[str, Any]],
) -> list[EvolutionExample]:
    return [
        (example.model_copy(deep=True) if isinstance(example, EvolutionExample)
         else EvolutionExample(**deepcopy(example)))
        for example in examples
    ]


def _run_examples(
    prompt: str,
    examples: Sequence[EvolutionExample],
    agent_factory: Callable[[str], Agent],
    task: TaskAdapter,
) -> list[EvaluationRecord]:
    agent = agent_factory(prompt)
    records: list[EvaluationRecord] = []
    for example in examples:
        inputs = task.collate(deepcopy(example.input))
        input_snapshot = deepcopy(inputs)
        agent_config = deepcopy(agent.to_config())
        message = agent.execute(inputs)
        prediction = task.prediction(message)
        metrics = task.evaluate(deepcopy(prediction), deepcopy(example.label))
        task.aggregate([metrics])  # Validate metric presence/finiteness before rewriting.
        records.append(
            EvaluationRecord(
                sample_id=example.sample_id,
                input=example.input,
                prediction=prediction,
                label=example.label,
                metrics=metrics,
                rendered_prompt=message.metadata.get("prompt"),
                agent_inputs=input_snapshot,
                agent_config=agent_config,
                response_metadata=deepcopy(message.metadata.get("llm", {})),
            )
        )
    return records


def _fingerprint(examples: Sequence[EvolutionExample]) -> str:
    return stable_hash(_json([{"sample_id": e.sample_id, "input": e.input, "label": e.label}
                              for e in examples]))
