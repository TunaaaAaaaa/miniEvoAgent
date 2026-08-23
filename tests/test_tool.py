import pytest

from evo_repro import Tool, ToolResult, get_word_length, get_word_length_tool


def test_tool_executes_function_and_returns_tool_result() -> None:
    result = get_word_length_tool.execute(
        {"text": "retrieval"},
        tool_call_id="call_1",
    )

    assert isinstance(result, ToolResult)
    assert result.tool_call_id == "call_1"
    assert result.name == "get_word_length"
    assert result.result == 9
    assert result.metadata["arguments"] == {"text": "retrieval"}


def test_tool_to_schema_returns_openai_function_schema() -> None:
    schema = get_word_length_tool.to_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_word_length"
    assert schema["function"]["description"]
    assert schema["function"]["parameters"]["required"] == ["text"]


def test_tool_argument_error_is_clear() -> None:
    with pytest.raises(ValueError, match="Invalid arguments for tool 'get_word_length'"):
        get_word_length_tool.execute({})


def test_tool_execution_error_is_not_swallowed() -> None:
    def explode() -> None:
        raise RuntimeError("boom")

    tool = Tool(
        name="explode",
        description="Raise an error.",
        parameters_schema={"type": "object", "properties": {}},
        function=explode,
    )

    with pytest.raises(RuntimeError, match="Tool 'explode' execution failed: boom"):
        tool.execute({})


def test_get_word_length_function() -> None:
    assert get_word_length("retrieval") == 9
