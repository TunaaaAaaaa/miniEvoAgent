from typing import Any

from pydantic import Field

from .base import BaseModel


class Message(BaseModel):
    """Final agent message returned to the caller."""

    content: str
    agent: str
    action: str
    metadata: dict[str, Any] = Field(default_factory=dict)
