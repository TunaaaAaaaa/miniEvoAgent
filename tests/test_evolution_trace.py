from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evo_repro import (
    Action, Agent, BaseLLM, EvolutionRoundTrace, GSM8KTask, LLMResponse,
    NQTask, OpenAILLM, PromptRewriter, PromptTemplate, RunProvenance,
    TaskAdapter, TextOutputParser, evolve_once,
)


class OfflineLLM(BaseLLM):
    def generate(self, prompt, messages=None, tools=None):
        answer = "wrong" if prompt.startswith("P0") else "42"
        return LLMResponse(content=answer, metadata={"response_id": "offline-response"})


class StatelessOptimizer(BaseLLM):
    def generate(self, prompt, messages=None, tools=None):
        return LLMResponse(content="  P1  ", metadata={"response_id": "rewrite-1"})


def factory(prompt):
    return Agent(name="test", description="offline", action=Action(
        name="answer", prompt_template=PromptTemplate(template=prompt + "\n{problem}"),
        llm=OfflineLLM(), output_parser=TextOutputParser(),
    ))


def run(task, label, **kwargs):
    return evolve_once(
        parent_prompt="P0", train_examples=[{"sample_id": "train", "input": "6*7?",
                                            "label": deepcopy(label)}],
        dev_examples=[{"sample_id": "dev", "input": "21+21?", "label": deepcopy(label)}],
        agent_factory=factory, rewriter=PromptRewriter(optimizer_llm=StatelessOptimizer()),
        task=task, **kwargs,
    )


@pytest.mark.parametrize("task,label,key", [
    (NQTask(input_key="problem"), ["forty-two", "42"], "f1"),
    (GSM8KTask(input_key="problem"), "21+21=42\n#### 42", "solve_rate"),
])
def test_two_metric_systems_complete_evolution_and_roundtrip(task, label, key):
    context = RunProvenance(git_sha="test-sha", git_dirty=False, seed=17,
                            dataset_name=task.name, dataset_revision="fixture-v1")
    trace = run(task, label, provenance=context)
    assert trace.score_key == key
    assert trace.parent_dev_score == 0.0
    assert trace.candidate_dev_score == 1.0
    assert trace.selection.accepted
    assert trace.rewrite_input.optimizer_meta_prompt
    assert "21+21?" not in trace.rewrite_input.optimizer_meta_prompt
    assert trace.rewrite_input.response_metadata == {"response_id": "rewrite-1"}
    assert trace.rewrite_input.response_content == "  P1  "
    assert trace.rewrite_input.optimizer_llm_config == {"class_name": "StatelessOptimizer"}
    assert trace.candidate_dev_records[0].agent_inputs == {"problem": "21+21?"}
    assert trace.candidate_dev_records[0].agent_config["action"]["llm_config"] == {
        "class_name": "OfflineLLM",
    }
    assert trace.candidate_dev_records[0].response_metadata["response_id"] == "offline-response"
    assert trace.provenance.seed == 17
    context.seed = 88
    assert trace.provenance.seed == 17
    assert EvolutionRoundTrace.from_json(trace.to_json()) == trace
    repeated = run(task, label)
    assert trace.train_fingerprint == repeated.train_fingerprint
    assert trace.dev_fingerprint == repeated.dev_fingerprint
    assert trace.train_fingerprint != trace.dev_fingerprint
    changed = run(task, ["43"] if key == "f1" else "#### 43")
    assert changed.dev_fingerprint != trace.dev_fingerprint


def test_openai_rewriter_records_actual_request_and_response_without_last_prompt(monkeypatch):
    import openai

    response = SimpleNamespace(
        id="completion-123", model="returned-model", system_fingerprint="fp_test",
        usage=SimpleNamespace(model_dump=lambda **kw: {"total_tokens": 9}),
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
            content="P1", tool_calls=None,
        ))],
    )
    create = Mock(return_value=response)
    monkeypatch.setattr(openai, "OpenAI", Mock(return_value=SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )))
    llm = OpenAILLM(model="requested-model", api_key="test-secret",
                    base_url="https://example.com/v1?api_key=hidden-secret",
                    temperature=0, top_p=1, seed=7, max_completion_tokens=64)
    trace = evolve_once(
        parent_prompt="P0", train_examples=[{"sample_id": "t", "input": "training",
                                            "label": "#### 42"}],
        dev_examples=[{"sample_id": "d", "input": "holdout", "label": "#### 42"}],
        agent_factory=factory, rewriter=PromptRewriter(optimizer_llm=llm),
        task=GSM8KTask(input_key="problem"),
    )
    sent = create.call_args.kwargs
    assert sent["seed"] == 7 and sent["temperature"] == 0
    assert sent["max_completion_tokens"] == 64
    assert trace.rewrite_input.optimizer_meta_prompt == sent["messages"][0]["content"]
    assert trace.rewrite_input.response_metadata["model"] == "returned-model"
    assert trace.rewrite_input.response_metadata["system_fingerprint"] == "fp_test"
    assert trace.rewrite_input.optimizer_llm_config["generation_parameters"]["seed"] == 7
    assert "test-secret" not in trace.to_json()
    assert "hidden-secret" not in trace.to_json()
    json.loads(trace.to_json())


def test_custom_structured_task_does_not_require_question_or_f1():
    class StructuredTask(TaskAdapter):
        name: str = "structured"
        score_key: str = "correct"

        def collate(self, value):
            return {"problem": value["text"]}

        def evaluate(self, prediction, label):
            return {"correct": float(prediction == label["expected"])}

    trace = evolve_once(
        parent_prompt="P0",
        train_examples=[{"sample_id": "t", "input": {"text": "train"},
                         "label": {"expected": "42"}}],
        dev_examples=[{"sample_id": "d", "input": {"text": "dev"},
                       "label": {"expected": "42"}}],
        agent_factory=factory, rewriter=PromptRewriter(optimizer_llm=StatelessOptimizer()),
        task=StructuredTask(),
    )
    assert trace.candidate_dev_score == 1.0
    assert trace.score_key == "correct"
    assert '"expected": "42"' in trace.rewrite_input.optimizer_meta_prompt


@pytest.mark.parametrize("train,dev", [
    ([], [{"sample_id": "d", "input": "q", "label": "a"}]),
    ([{"sample_id": "t", "input": "q", "label": "a"}], []),
    ([{"sample_id": "same", "input": "q", "label": "a"}],
     [{"sample_id": "same", "input": "q", "label": "a"}]),
])
def test_invalid_splits_fail_before_any_model_call(train, dev):
    never = Mock(side_effect=AssertionError("must not run"))
    with pytest.raises(ValueError):
        evolve_once(parent_prompt="P0", train_examples=train, dev_examples=dev,
                    agent_factory=never, rewriter=PromptRewriter(optimizer_llm=StatelessOptimizer()))
    never.assert_not_called()
