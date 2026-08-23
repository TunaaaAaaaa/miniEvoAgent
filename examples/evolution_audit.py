from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evo_repro import (  # noqa: E402
    Action,
    Agent,
    BaseLLM,
    LLMResponse,
    PromptRewriter,
    PromptTemplate,
    TextOutputParser,
    evolve,
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


class PromptAwareFakeLLM(BaseLLM):
    def generate(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        question = prompt.split("Question:", 1)[1].strip()
        prompt_name = prompt.splitlines()[0].strip()
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


def agent_factory(system_prompt: str) -> Agent:
    return Agent(
        name="qa_agent",
        description="Answers deterministic QA examples.",
        action=Action(
            name="answer_question",
            prompt_template=PromptTemplate(
                template=f"{system_prompt}\n\nQuestion: {{question}}"
            ),
            llm=PromptAwareFakeLLM(),
            output_parser=TextOutputParser(),
        ),
    )


def print_record(title: str, record) -> None:
    print(title)
    print()
    print("ID:")
    print(record.sample_id)
    print()
    print("Input:")
    print(record.input)
    print()
    print("Prediction:")
    print(record.prediction)
    print()
    print("Ground Truth:" if "Train" in title else "Label:")
    print(record.label)
    print()
    print("Metrics:")
    print(format_metrics(record.metrics))
    print()


def format_metrics(metrics: dict[str, float]) -> str:
    return "\n".join(f"{name.upper()} = {value:.2f}" for name, value in metrics.items())


def print_records(records, label: str) -> None:
    for index, record in enumerate(records, start=1):
        print_record(f"[{label} Sample {index}]", record)


def main() -> None:
    initial_prompt = "P0"
    rewriter = PromptRewriter(
        optimizer_llm=SequentialRewriteLLM(["P1", "P2", "P3"])
    )
    final_prompt, history = evolve(
        initial_prompt=initial_prompt,
        train_examples=TRAIN_EXAMPLES,
        dev_examples=DEV_EXAMPLES,
        agent_factory=agent_factory,
        rewriter=rewriter,
        generations=3,
    )

    print("=" * 50)
    print("MINI SELF-EVOLVING AGENT AUDIT")
    print("=" * 50)
    print()
    print("Initial Prompt:")
    print(initial_prompt)
    print()

    for trace in history:
        print("=" * 50)
        print(f"GENERATION {trace.generation}")
        print("=" * 50)
        print()
        print("[PARENT PROMPT]")
        print()
        print(trace.parent_prompt)
        print()
        print("---------------- TRAIN EXECUTION ----------------")
        print()
        print_records(trace.train_records, "Train")
        print("---------------- REWRITE ----------------")
        print()
        print("Current Prompt:")
        print(trace.rewrite_input.current_prompt)
        print()
        print("Feedback Examples:")
        for example in trace.rewrite_input.feedback_examples:
            print(f"- ID: {example.sample_id}")
            print(f"  Input: {example.input}")
            print(f"  Prediction: {example.prediction}")
            print(f"  Ground Truth: {example.label}")
            print("  Metrics:")
            for name, value in (example.metrics or {}).items():
                print(f"    {name.upper()} = {value:.2f}")
        print()
        print("===== REWRITER META PROMPT =====")
        print(trace.rewrite_input.optimizer_meta_prompt)
        print("===== END REWRITER META PROMPT =====")
        print()
        print("Candidate Prompt:")
        print(trace.candidate_prompt)
        print()
        print("---------------- DEV: PARENT ----------------")
        print()
        print("Prompt Used:")
        print(trace.parent_prompt_used)
        print()
        print_records(trace.parent_dev_records, "Dev")
        print("Parent Mean F1:")
        print(f"{trace.parent_dev_score:.2f}")
        print()
        print("---------------- DEV: CANDIDATE ----------------")
        print()
        print("Prompt Used:")
        print(trace.candidate_prompt_used)
        print()
        print_records(trace.candidate_dev_records, "Dev")
        print("Candidate Mean F1:")
        print(f"{trace.candidate_dev_score:.2f}")
        print()
        print("---------------- SELECTION ----------------")
        print()
        print("Parent Score:")
        print(f"{trace.selection.parent_score:.2f}")
        print()
        print("Candidate Score:")
        print(f"{trace.selection.candidate_score:.2f}")
        print()
        print("Decision:")
        print("ACCEPT CANDIDATE" if trace.selection.accepted else "REJECT CANDIDATE")
        print()
        print("Selected Prompt:")
        print(trace.selection.selected_prompt)
        print()

    print("=" * 50)
    print("EVOLUTION SUMMARY")
    print("=" * 50)
    print()
    for trace in history:
        decision = "ACCEPT" if trace.selection.accepted else "REJECT"
        print(f"Generation {trace.generation}")
        print(f"{trace.parent_prompt} -> {trace.candidate_prompt}")
        print(f"{trace.parent_dev_score:.2f} -> {trace.candidate_dev_score:.2f}")
        print(decision)
        if not trace.selection.accepted:
            print(f"Survivor = {trace.selection.selected_prompt}")
        print()
    print("Initial Prompt:")
    print(initial_prompt)
    print()
    print("Final Prompt:")
    print(final_prompt)
    print()
    print("Evolution completed successfully.")


if __name__ == "__main__":
    main()
