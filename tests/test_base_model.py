import json
from typing import Any

from evo_repro import (
    Action,
    ActionOutput,
    Agent,
    BaseLLM,
    BaseModel,
    EvolutionExample,
    LLMResponse,
    Message,
    OpenAILLM,
    PromptRewriter,
    PromptTemplate,
    TextOutputParser,
    Tool,
    ToolResult,
)


class FakeLLM(BaseLLM):
    def generate(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        return LLMResponse(content="ok")


def test_base_model_adds_class_name_and_version() -> None:
    template = PromptTemplate(template="Question: {question}")

    assert isinstance(template, BaseModel)
    assert template.class_name == "PromptTemplate"
    assert template.version == 0


def test_base_model_round_trips_through_dict_and_json() -> None:
    message = Message(
        content="Paris",
        agent="qa_agent",
        action="answer_question",
        metadata={"score": 1.0},
    )

    restored_from_dict = Message.from_dict(message.to_dict())
    restored_from_json = Message.from_json(message.to_json())

    assert restored_from_dict == message
    assert restored_from_json == message
    assert restored_from_json.class_name == "Message"


def test_base_model_saves_and_loads_json_file(tmp_path) -> None:
    example = EvolutionExample(
        sample_id="dev_001",
        input="Capital of France?",
        label="Paris",
    )

    path = example.save_json(tmp_path / "example.json")

    assert EvolutionExample.from_json_file(path) == example


def test_existing_data_models_share_common_base() -> None:
    models = [
        ActionOutput(content="ok"),
        ToolResult(tool_call_id="call_1", name="lookup", result={"ok": True}),
        PromptTemplate(template="{question}"),
    ]

    assert all(isinstance(model, BaseModel) for model in models)


def test_runtime_action_config_excludes_runtime_llm_object() -> None:
    action = Action(
        name="answer_question",
        prompt_template=PromptTemplate(template="Question: {question}"),
        llm=FakeLLM(),
        output_parser=TextOutputParser(),
        max_tool_rounds=2,
    )

    config = action.to_config()

    assert config["class_name"] == "Action"
    assert config["llm_config"] == {"class_name": "FakeLLM"}
    assert "llm" not in config
    assert config["prompt_template"]["template"] == "Question: {question}"
    assert config["output_parser"]["class_name"] == "TextOutputParser"
    assert config["max_tool_rounds"] == 2
    json.dumps(config)


def test_tool_config_uses_function_reference_not_function_object() -> None:
    def lookup(value: str) -> str:
        return value

    tool = Tool(
        name="lookup",
        description="Return a value.",
        parameters_schema={"type": "object", "properties": {}},
        function=lookup,
    )

    config = tool.to_config()

    assert "function" not in config
    assert config["function_ref"]["qualname"].endswith("lookup")
    json.dumps(config)


def test_agent_and_rewriter_configs_keep_runtime_dependencies_as_configs() -> None:
    action = Action(
        name="answer_question",
        prompt_template=PromptTemplate(template="{question}"),
        llm=FakeLLM(),
        output_parser=TextOutputParser(),
    )
    agent = Agent(name="qa_agent", description="QA", action=action)
    rewriter = PromptRewriter(optimizer_llm=FakeLLM())

    agent_config = agent.to_config()
    rewriter_config = rewriter.to_config()

    assert agent_config["action"]["llm_config"] == {"class_name": "FakeLLM"}
    assert rewriter_config["optimizer_llm_config"] == {"class_name": "FakeLLM"}
    assert "optimizer_llm" not in rewriter_config
    json.dumps(agent_config)
    json.dumps(rewriter_config)


def test_openai_config_redacts_secrets_embedded_in_base_url() -> None:
    llm = OpenAILLM(
        api_key="provider-key",
        base_url="https://user:password@example.com/v1?api_key=query-secret",
    )

    config = llm.to_config()

    assert config["base_url"] == "https://user:***@example.com/v1?api_key=***"
    assert "password" not in json.dumps(config)
    assert "query-secret" not in json.dumps(config)
