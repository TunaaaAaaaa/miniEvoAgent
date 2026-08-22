from pydantic import BaseModel

from .llms import BaseLLM
from .parsers import ActionOutput, TextOutputParser
from .prompts import PromptTemplate


class Action(BaseModel):
    """One executable task: render prompt, call LLM, parse output."""

    name: str
    prompt_template: PromptTemplate
    llm: BaseLLM
    output_parser: TextOutputParser

    model_config = {"arbitrary_types_allowed": True}

    def execute(self, inputs: dict[str, object]) -> ActionOutput:
        prompt = self.prompt_template.format(**inputs)
        llm_response = self.llm.generate(prompt)
        return self.output_parser.parse(llm_response)
