from collections.abc import Callable, Sequence
import random
from statistics import mean
from typing import Any

from pydantic import Field

from .agents import Agent
from .base import BaseModel
from .messages import Message
from .storage import PostgreSQLStorage


CollateFunc = Callable[[Any], dict[str, Any]]
PostprocessFunc = Callable[[Message], Any]


class AgentEvaluationRecord(BaseModel):
    """Per-example result from evaluating an agent on a benchmark."""

    benchmark_name: str
    split: str
    sample_id: str
    input: Any
    prediction: Any
    label: Any
    metrics: dict[str, float]
    rendered_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentEvaluationResult(BaseModel):
    """Aggregate result plus per-example records for an agent evaluation."""

    benchmark_name: str
    split: str
    records: list[AgentEvaluationRecord]
    aggregate_metrics: dict[str, float]
    score_key: str = "f1"

    @property
    def score(self) -> float:
        if self.score_key not in self.aggregate_metrics:
            raise KeyError(
                f"Score key '{self.score_key}' not found in aggregate metrics: "
                f"{sorted(self.aggregate_metrics)}"
            )
        return self.aggregate_metrics[self.score_key]


class Evaluator(BaseModel):
    """Evaluate miniEvoAgent agents against EvoAgentX-style benchmarks."""

    score_key: str = "f1"

    def evaluate_agent(
        self,
        *,
        agent: Agent,
        benchmark: Any,
        split: str = "dev",
        collate_func: CollateFunc,
        output_postprocess_func: PostprocessFunc | None = None,
        sample_k: int | None = None,
        seed: int | None = None,
        storage: PostgreSQLStorage | None = None,
        run_id: int | None = None,
        prompt_id: int | None = None,
        benchmark_task_type: str | None = None,
    ) -> AgentEvaluationResult:
        """Run an agent on one benchmark split and optionally persist results."""

        examples = self._get_examples(
            benchmark=benchmark,
            split=split,
            sample_k=sample_k,
            seed=seed,
        )
        output_postprocess_func = output_postprocess_func or (lambda message: message.content)
        benchmark_name = self._get_benchmark_name(benchmark)

        benchmark_storage_id = None
        example_storage_ids: dict[str, int] = {}
        if storage is not None:
            benchmark_storage_id = storage.upsert_benchmark(
                name=benchmark_name,
                task_type=benchmark_task_type or getattr(benchmark, "task_type", "unknown"),
                description=getattr(benchmark, "description", None),
                metadata={"source": type(benchmark).__name__},
            )

        records: list[AgentEvaluationRecord] = []
        for example in examples:
            sample_id = str(benchmark.get_id(example))
            label = benchmark.get_label(example)
            agent_inputs = collate_func(example)
            if not isinstance(agent_inputs, dict):
                raise ValueError(
                    "collate_func must return a dict suitable for Agent.execute()."
                )

            message = agent.execute(agent_inputs)
            prediction = output_postprocess_func(message)
            metrics = benchmark.evaluate(prediction=prediction, label=label)
            rendered_prompt = message.metadata.get("prompt")

            record = AgentEvaluationRecord(
                benchmark_name=benchmark_name,
                split=split,
                sample_id=sample_id,
                input=agent_inputs,
                prediction=prediction,
                label=label,
                metrics=metrics,
                rendered_prompt=rendered_prompt,
                metadata={"agent": agent.name, "action": agent.action.name},
            )
            records.append(record)

            if storage is not None and benchmark_storage_id is not None:
                example_storage_ids[sample_id] = storage.add_example(
                    benchmark_id=benchmark_storage_id,
                    split=split,
                    sample_id=sample_id,
                    input_data=agent_inputs,
                    label=label,
                    metadata={"source": type(benchmark).__name__},
                )

        result = AgentEvaluationResult(
            benchmark_name=benchmark_name,
            split=split,
            records=records,
            aggregate_metrics=self._aggregate_metrics(records),
            score_key=self.score_key,
        )

        if storage is not None:
            evaluation_id = storage.save_evaluation(
                run_id=run_id,
                prompt_id=prompt_id,
                split=split,
                aggregate_metrics=result.aggregate_metrics,
            )
            for record in records:
                storage.save_evaluation_item(
                    evaluation_id=evaluation_id,
                    example_id=example_storage_ids.get(record.sample_id),
                    prediction=record.prediction,
                    label=record.label,
                    metrics=record.metrics,
                    rendered_prompt=record.rendered_prompt,
                    metadata=record.metadata,
                )

        return result

    def _get_examples(
        self,
        *,
        benchmark: Any,
        split: str,
        sample_k: int | None,
        seed: int | None,
    ) -> list[Any]:
        if split not in {"train", "dev", "test"}:
            raise ValueError("split must be one of: train, dev, test")

        method_name = f"get_{split}_data"
        if hasattr(benchmark, method_name):
            method = getattr(benchmark, method_name)
            try:
                return list(method(sample_k=sample_k, seed=seed))
            except TypeError:
                examples = list(method())
        elif hasattr(benchmark, "get_data_by_mode"):
            examples = list(benchmark.get_data_by_mode(mode=split))
        else:
            raise TypeError(
                "benchmark must provide EvoAgentX-style split accessors such as "
                "get_train_data/get_dev_data/get_test_data."
            )

        return self._sample_examples(examples, sample_k=sample_k, seed=seed)

    def _sample_examples(
        self,
        examples: Sequence[Any],
        *,
        sample_k: int | None,
        seed: int | None,
    ) -> list[Any]:
        examples = list(examples)
        if sample_k is None:
            return examples
        if sample_k < 0:
            raise ValueError("sample_k must be non-negative.")
        rng = random.Random(seed)
        return rng.sample(examples, k=min(sample_k, len(examples)))

    def _aggregate_metrics(
        self,
        records: Sequence[AgentEvaluationRecord],
    ) -> dict[str, float]:
        if not records:
            raise ValueError("Cannot aggregate an empty evaluation result.")

        metric_names = sorted(
            {
                name
                for record in records
                for name, value in record.metrics.items()
                if isinstance(value, (int, float))
            }
        )
        return {
            name: mean(float(record.metrics[name]) for record in records if name in record.metrics)
            for name in metric_names
        }

    def _get_benchmark_name(self, benchmark: Any) -> str:
        return str(getattr(benchmark, "name", type(benchmark).__name__))
