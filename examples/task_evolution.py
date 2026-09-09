"""Offline NQ/GSM8K-shaped smoke checks. Scores are synthetic, not benchmark results."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evo_repro import (  # noqa: E402
    Action, Agent, BaseLLM, GSM8KTask, LLMResponse, NQTask, PromptRewriter,
    PromptTemplate, RunProvenance, TextOutputParser, evolve_once,
)


FIXTURES = {
    "nq": {
        "train": {"sample_id": "nq-train-1", "input": "Capital of France?",
                  "label": ["Paris", "the city of Paris"]},
        "dev": {"sample_id": "nq-dev-1", "input": "Capital of Japan?",
                "label": ["Tokyo", "Tokyo Metropolis"]},
        "answers": {"Capital of France?": "Paris", "Capital of Japan?": "Tokyo"},
    },
    "gsm8k": {
        "train": {"sample_id": "gsm-train-1", "input": "3 boxes of 4 pencils. How many?",
                  "label": "3 * 4 = 12\n#### 12"},
        "dev": {"sample_id": "gsm-dev-1", "input": "5 bags of 6 apples. How many?",
                "label": "5 * 6 = 30\n#### 30"},
        "answers": {"3 boxes of 4 pencils. How many?": "#### 12",
                    "5 bags of 6 apples. How many?": "#### 30"},
    },
}


class FixtureLLM(BaseLLM):
    def __init__(self, answers):
        self.answers = answers

    def generate(self, prompt, messages=None, tools=None):
        question = prompt.split("Input: ", 1)[1]
        content = self.answers[question] if prompt.startswith("P1") else "unknown"
        return LLMResponse(content=content, metadata={"provider": "offline-fixture"})


class FixtureOptimizer(BaseLLM):
    def generate(self, prompt, messages=None, tools=None):
        return LLMResponse(content="P1", metadata={"provider": "offline-fixture"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "nq", "gsm8k"], default="all")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for name in (["nq", "gsm8k"] if args.task == "all" else [args.task]):
        fixture = FIXTURES[name]

        def factory(prompt):
            return Agent(name="fixture-agent", description="Synthetic smoke check", action=Action(
                name="answer", prompt_template=PromptTemplate(template=prompt + "\nInput: {task}"),
                llm=FixtureLLM(fixture["answers"]), output_parser=TextOutputParser(),
            ))

        task = NQTask(input_key="task") if name == "nq" else GSM8KTask(input_key="task")
        trace = evolve_once(
            parent_prompt="P0", train_examples=[fixture["train"]], dev_examples=[fixture["dev"]],
            agent_factory=factory, rewriter=PromptRewriter(optimizer_llm=FixtureOptimizer()),
            task=task, provenance=RunProvenance.capture(
                dataset_name=f"synthetic-{name}", dataset_revision="fixture-v1",
            ),
        )
        assert trace.selection.accepted and trace.candidate_dev_score == 1.0
        print(f"{name}: {trace.score_key} {trace.parent_dev_score} -> "
              f"{trace.candidate_dev_score} (synthetic smoke check)")
        if args.output_dir:
            trace.save_json(args.output_dir / f"{name}-trace.json")


if __name__ == "__main__":
    main()
