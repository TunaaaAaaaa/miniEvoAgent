from pydantic import BaseModel

from .actions import Action
from .messages import Message


class Agent(BaseModel):
    """Minimal single-action agent."""

    name: str
    description: str
    action: Action

    model_config = {"arbitrary_types_allowed": True}

    def execute(self, inputs: dict[str, object]) -> Message:
        output = self.action.execute(inputs)
        return Message(
            content=output.content,
            agent=self.name,
            action=self.action.name,
            metadata={"description": self.description},
        )
