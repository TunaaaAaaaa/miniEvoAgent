from typing import Any

import pytest

from evo_repro import (
    Action,
    Agent,
    AgentEvaluationResult,
    BaseLLM,
    Evaluator,
    LLMResponse,
    PromptTemplate,
    TextOutputParser,
)


class FakeBenchmark:
    name = "fake_evoagentx_qa"
    task_type = "qa"
    description = "EvoAgentX-style fake benchmark."

    def __init__(self) -> None:
        self.train = [
            {"id": "train-1", "question": "Training color?", "answer": "orange"}
        ]
        self.dev = [
            {"id": "dev-1", "question": "Capital of France?", "answer": "Paris"},
            {"id": "dev-2", "question": "Largest planet?", "answer": "Jupiter"},
        ]
        self.test = [
            {"id": "test-1", "question": "Author of Hamlet?", "answer": "Shakespeare"}
        ]

    def get_train_data(self, sample_k=None, seed=None):
        return self.train[:sample_k] if sample_k else self.train

    def get_dev_data(self, sample_k=None, seed=None):
        return self.dev[:sample_k] if sample_k else self.dev

    def get_test_data(self, sample_k=None, seed=None):
        return self.test[:sample_k] if sample_k else self.test

    def get_id(self, example: dict[str, str]) -> str:
        return example["id"]

    def get_label(self, example: dict[str, str]) -> str:
        return example["answer"]

    def evaluate(self, prediction: Any, label: Any) -> dict[str, float]:
        return {
            "f1": 1.0 if str(label).lower() in str(prediction).lower() else 0.0,
            "em": 1.0 if prediction == label else 0.0,
        }


class AnsweringLLM(BaseLLM):
    def generate(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        answers = {
            "Capital of France?": "Final answer: Paris",
            "Largest planet?": "Final answer: Jupiter",
            "Author of Hamlet?": "Final answer: Shakespeare",
        }
        question = prompt.split("Question:", 1)[1].strip()
        return LLMResponse(content=answers.get(question, "Final answer: unknown"))


class RecordingStorage:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def upsert_benchmark(self, **kwargs) -> int:
        self.calls.append(("upsert_benchmark", kwargs))
        return 10

    def add_example(self, **kwargs) -> int:
        self.calls.append(("add_example", kwargs))
        return 20 + len([name for name, _ in self.calls if name == "add_example"])

    def save_evaluation(self, **kwargs) -> int:
        self.calls.append(("save_evaluation", kwargs))
        return 30

    def save_evaluation_item(self, **kwargs) -> int:
        self.calls.append(("save_evaluation_item", kwargs))
        return 40


def build_agent() -> Agent:
    return Agent(
        name="qa_agent",
        description="Answers benchmark examples.",
        action=Action(
            name="answer_question",
            prompt_template=PromptTemplate(template="Question: {question}"),
            llm=AnsweringLLM(),
            output_parser=TextOutputParser(),
        ),
    )


def test_evaluator_runs_evoagentx_style_benchmark() -> None:
    evaluator = Evaluator(score_key="f1")

    result = evaluator.evaluate_agent(
        agent=build_agent(),
        benchmark=FakeBenchmark(),
        split="dev",
        collate_func=lambda example: {"question": example["question"]},
        output_postprocess_func=lambda message: (
            message.content.split("Final answer:", 1)[1].strip()
        ),
    )

    assert isinstance(result, AgentEvaluationResult)
    assert result.benchmark_name == "fake_evoagentx_qa"
    assert result.split == "dev"
    assert result.score == 1.0
    assert result.aggregate_metrics == {"em": 1.0, "f1": 1.0}
    assert [record.sample_id for record in result.records] == ["dev-1", "dev-2"]
    assert result.records[0].rendered_prompt == "Question: Capital of France?"


def test_evaluator_requires_collate_func_to_return_dict() -> None:
    evaluator = Evaluator()

    with pytest.raises(ValueError, match="collate_func must return a dict"):
        evaluator.evaluate_agent(
            agent=build_agent(),
            benchmark=FakeBenchmark(),
            split="dev",
            collate_func=lambda example: "not a dict",
        )


def test_evaluator_rejects_unknown_split() -> None:
    evaluator = Evaluator()

    with pytest.raises(ValueError, match="split must be one of"):
        evaluator.evaluate_agent(
            agent=build_agent(),
            benchmark=FakeBenchmark(),
            split="validation",
            collate_func=lambda example: {"question": example["question"]},
        )


def test_evaluator_persists_records_when_storage_is_provided() -> None:
    storage = RecordingStorage()
    evaluator = Evaluator(score_key="f1")

    result = evaluator.evaluate_agent(
        agent=build_agent(),
        benchmark=FakeBenchmark(),
        split="dev",
        collate_func=lambda example: {"question": example["question"]},
        output_postprocess_func=lambda message: (
            message.content.split("Final answer:", 1)[1].strip()
        ),
        storage=storage,
        run_id=7,
        prompt_id=8,
    )

    call_names = [name for name, _ in storage.calls]
    assert call_names.count("upsert_benchmark") == 1
    assert call_names.count("add_example") == 2
    assert call_names.count("save_evaluation") == 1
    assert call_names.count("save_evaluation_item") == 2
    assert result.score == 1.0
    assert storage.calls[0][1]["name"] == "fake_evoagentx_qa"
    assert storage.calls[-1][1]["evaluation_id"] == 30
