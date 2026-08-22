from evo_repro import Action, Agent, OpenAILLM, PromptTemplate, TextOutputParser


def build_agent() -> Agent:
    llm = OpenAILLM()
    action = Action(
        name="answer_question",
        prompt_template=PromptTemplate(
            template="Answer the following question:\n\n{question}"
        ),
        llm=llm,
        output_parser=TextOutputParser(),
    )
    return Agent(
        name="qa_agent",
        description="Answers a single user question.",
        action=action,
    )


def main() -> None:
    agent = build_agent()
    message = agent.execute(
        {"question": "What is Retrieval-Augmented Generation?"}
    )

    print("Agent:")
    print(message.agent)
    print()
    print("Action:")
    print(message.action)
    print()
    print("Answer:")
    print(message.content)


if __name__ == "__main__":
    main()
