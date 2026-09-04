"""Evaluate a miniEvoAgent agent on an EvoAgentX benchmark.

Set PYTHONPATH to the EvoAgentX repo if it is not installed:

    $env:PYTHONPATH="D:\\Projects\\RSIProject\\reference\\EvoAgentX"
"""

from evo_repro import Action, Agent, Evaluator, OpenAILLM, PromptTemplate, TextOutputParser


def main() -> None:
    try:
        from evoagentx.benchmark import NQ
    except ImportError as exc:
        raise SystemExit(
            "Install EvoAgentX or set PYTHONPATH to the EvoAgentX reference repo."
        ) from exc

    benchmark = NQ(mode="dev")
    agent = Agent(
        name="qa_agent",
        description="Answers Natural Questions examples.",
        action=Action(
            name="answer_question",
            prompt_template=PromptTemplate(
                template="Answer the question concisely.\n\nQuestion: {question}"
            ),
            llm=OpenAILLM(),
            output_parser=TextOutputParser(),
        ),
    )

    result = Evaluator(score_key="f1").evaluate_agent(
        agent=agent,
        benchmark=benchmark,
        split="dev",
        sample_k=5,
        seed=42,
        collate_func=lambda example: {"question": example["question"]},
    )
    print(result.aggregate_metrics)


if __name__ == "__main__":
    main()
