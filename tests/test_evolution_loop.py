from statistics import mean
from typing import Any

from evo_repro import (
    Action,
    Agent,
    BaseLLM,
    LLMResponse,
    PromptRewriter,
    PromptTemplate,
    TextOutputParser,
    evolve,
    evolve_once,
)


TRAIN_EXAMPLES = [
    {"sample_id": "train_001", "input": "Training color?", "label": "orange"},
    {"sample_id": "train_002", "input": "Training animal?", "label": "otter"},
]

DEV_EXAMPLES = [
    {"sample_id": "dev_001", "input": "Capital of France?", "label": "Paris"},
    {"sample_id": "dev_002", "input": "Author of Hamlet?", "label": "William Shakespeare"},
    {"sample_id": "dev_003", "input": "Largest planet?", "label": "Jupiter"},
    {"sample_id": "dev_004", "input": "Fastest land animal?", "label": "cheetah"},
]


class PromptAwareFakeLLM(BaseLLM):
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def generate(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        question = prompt.split("Question:", 1)[1].strip()
        prompt_name = prompt.splitlines()[0].strip()
        self.calls.append({"prompt_name": prompt_name, "question": question})
        answer = ANSWERS_BY_PROMPT[prompt_name].get(question, "wrong")
        return LLMResponse(content=answer, metadata={"provider": "fake-agent"})


class SequentialRewriteLLM(BaseLLM):
    def __init__(self, candidates: list[str]) -> None:
        self.candidates = candidates
        self.call_count = 0
        self.last_prompt: str | None = None

    def generate(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        self.last_prompt = prompt
        candidate = self.candidates[self.call_count]
        self.call_count += 1
        return LLMResponse(content=candidate)


ANSWERS_BY_PROMPT = {
    "P0": {
        "Training color?": "orange",
        "Training animal?": "wrong",
        "Capital of France?": "Paris",
        "Author of Hamlet?": "William Shakespeare",
        "Largest planet?": "wrong",
        "Fastest land animal?": "wrong",
    },
    "P1": {
        "Training color?": "orange",
        "Training animal?": "otter",
        "Capital of France?": "Paris",
        "Author of Hamlet?": "William Shakespeare",
        "Largest planet?": "Jupiter",
        "Fastest land animal?": "wrong",
    },
    "P2": {
        "Capital of France?": "Paris",
        "Author of Hamlet?": "wrong",
        "Largest planet?": "wrong",
        "Fastest land animal?": "wrong",
    },
    "P3": {
        "Capital of France?": "Paris",
        "Author of Hamlet?": "William Shakespeare",
        "Largest planet?": "Jupiter",
        "Fastest land animal?": "cheetah",
    },
}


def make_agent_factory(fake_llm: PromptAwareFakeLLM):
    def agent_factory(system_prompt: str) -> Agent:
        return Agent(
            name="qa_agent",
            description="Answers deterministic QA examples.",
            action=Action(
                name="answer_question",
                prompt_template=PromptTemplate(
                    template=f"{system_prompt}\n\nQuestion: {{question}}"
                ),
                llm=fake_llm,
                output_parser=TextOutputParser(),
            ),
        )

    return agent_factory


def test_evolve_once_re_evaluates_candidate_on_dev_set() -> None:
    fake_llm = PromptAwareFakeLLM()
    rewriter = PromptRewriter(optimizer_llm=SequentialRewriteLLM(["P1"]))

    trace = evolve_once(
        parent_prompt="P0",
        train_examples=TRAIN_EXAMPLES,
        dev_examples=DEV_EXAMPLES,
        agent_factory=make_agent_factory(fake_llm),
        rewriter=rewriter,
    )

    assert trace.candidate_prompt == "P1"
    assert trace.selection.accepted is True
    assert trace.parent_dev_score == 0.5
    assert trace.candidate_dev_score == 0.75
    assert [record.sample_id for record in trace.parent_dev_records] == [
        record.sample_id for record in trace.candidate_dev_records
    ]
    assert trace.dev_sample_ids == ["dev_001", "dev_002", "dev_003", "dev_004"]
    assert trace.candidate_dev_score == mean(
        record.metrics["f1"] for record in trace.candidate_dev_records
    )
    assert all("P1" in record.rendered_prompt for record in trace.candidate_dev_records)
    assert any(call["prompt_name"] == "P1" for call in fake_llm.calls)


def test_evolve_three_generations_and_trace_invariants() -> None:
    fake_llm = PromptAwareFakeLLM()
    rewriter = PromptRewriter(optimizer_llm=SequentialRewriteLLM(["P1", "P2", "P3"]))

    final_prompt, history = evolve(
        initial_prompt="P0",
        train_examples=TRAIN_EXAMPLES,
        dev_examples=DEV_EXAMPLES,
        agent_factory=make_agent_factory(fake_llm),
        rewriter=rewriter,
        generations=3,
    )

    assert [trace.selection.accepted for trace in history] == [True, False, True]
    assert history[1].parent_prompt == history[0].selected_prompt
    assert history[2].parent_prompt == history[1].selected_prompt
    assert history[1].candidate_prompt == "P2"
    assert history[2].parent_prompt == "P1"
    assert final_prompt == history[2].selected_prompt == "P3"

    for trace in history:
        assert trace.candidate_prompt != ""
        assert [record.sample_id for record in trace.parent_dev_records] == [
            record.sample_id for record in trace.candidate_dev_records
        ]
        assert trace.candidate_dev_score == mean(
            record.metrics["f1"] for record in trace.candidate_dev_records
        )
        assert all(
            trace.candidate_prompt in record.rendered_prompt
            for record in trace.candidate_dev_records
        )
        train_ids = set(trace.rewrite_input.rewriter_training_sample_ids)
        dev_ids = set(trace.dev_sample_ids)
        assert train_ids.isdisjoint(dev_ids)
        assert trace.rewrite_input.optimizer_meta_prompt is not None
        for dev_example in DEV_EXAMPLES:
            assert dev_example["label"] not in trace.rewrite_input.optimizer_meta_prompt

    assert history[0].candidate_dev_score > history[0].parent_dev_score
    assert history[0].selection.accepted is True
    assert history[1].candidate_dev_score <= history[1].parent_dev_score
    assert history[1].selection.accepted is False
    assert history[2].candidate_dev_score > history[2].parent_dev_score
    assert history[2].selection.accepted is True
