import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evo_repro import Action, Agent, OpenAILLM, PromptTemplate, TextOutputParser
from evo_repro import get_word_length_tool


def build_agent() -> Agent:
    llm = OpenAILLM()
    action = Action(
        name="answer_with_tools",
        prompt_template=PromptTemplate(
            template="Answer the following question. Use tools when useful:\n\n{question}"
        ),
        llm=llm,
        output_parser=TextOutputParser(),
        tools=[get_word_length_tool],
    )
    return Agent(
        name="tool_agent",
        description="Answers a question and can use tools.",
        action=action,
    )


def main() -> None:
    question = 'How many characters are in the word "retrieval"?'
    agent = build_agent()

    print("User Input")
    print(question)
    print()
    print("Prompt")
    print(agent.action.prompt_template.format(question=question))
    print()

    message = agent.execute({"question": question})

    tool_results = message.metadata["tool_results"]
    if tool_results:
        print("LLM ToolCall")
        for tool_result in tool_results:
            print(f"{tool_result['tool_call_id']} -> {tool_result['name']}")
        print()
        print("Tool Execution")
        print("get_word_length(text='retrieval')")
        print()
        print("ToolResult")
        print(tool_results)
        print()
        print("Second LLM Call")
        print(f"tool_rounds={message.metadata['tool_rounds']}")
        print()
    else:
        print("LLM ToolCall")
        print("None")
        print()

    print("Final LLMResponse")
    print(message.content)
    print()
    print("ActionOutput")
    print(message.content)
    print()
    print("Agent Message")
    print(message.content)


if __name__ == "__main__":
    main()
