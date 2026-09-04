from typing import Any

from .actions import Action
from .base import BaseModel
from .messages import Message


class Agent(BaseModel):
    """Minimal single-action agent."""

    name: str
    description: str
    action: Action

    model_config = {"arbitrary_types_allowed": True}

    def execute(self, inputs: dict[str, object]) -> Message:
        output = self.action.execute(inputs)
        metadata = {"description": self.description}
        metadata.update(output.metadata)
        return Message(
            content=output.content,
            agent=self.name,
            action=self.action.name,
            metadata=metadata,
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "action": self.action.to_config(),
        }
