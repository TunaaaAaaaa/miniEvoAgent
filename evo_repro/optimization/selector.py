from pydantic import BaseModel


class SelectionResult(BaseModel):
    """Decision made by comparing one parent prompt with one candidate prompt."""

    selected_prompt: str
    parent_score: float
    candidate_score: float
    accepted: bool
    selected_source: str


class PromptSelector:
    """Deterministic 1-parent, 1-candidate selector."""

    def select(
        self,
        parent_prompt: str,
        parent_score: float,
        candidate_prompt: str,
        candidate_score: float,
    ) -> SelectionResult:
        accepted = candidate_score > parent_score
        return SelectionResult(
            selected_prompt=candidate_prompt if accepted else parent_prompt,
            parent_score=parent_score,
            candidate_score=candidate_score,
            accepted=accepted,
            selected_source="candidate" if accepted else "parent",
        )
