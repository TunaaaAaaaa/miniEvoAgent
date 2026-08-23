from .tool import Tool


def get_word_length(text: str) -> int:
    return len(text)


get_word_length_tool = Tool(
    name="get_word_length",
    description="Get the character length of a word or text.",
    parameters_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The word or text to measure.",
            }
        },
        "required": ["text"],
        "additionalProperties": False,
    },
    function=get_word_length,
)
