from evo_repro import PromptSelector


def test_selector_accepts_strictly_better_candidate() -> None:
    selector = PromptSelector()

    result = selector.select(
        parent_prompt="P0",
        parent_score=0.5,
        candidate_prompt="P1",
        candidate_score=0.75,
    )

    assert result.accepted is True
    assert result.selected_prompt == "P1"
    assert result.selected_source == "candidate"
    assert result.parent_score == 0.5
    assert result.candidate_score == 0.75


def test_selector_keeps_parent_on_tie() -> None:
    selector = PromptSelector()

    result = selector.select(
        parent_prompt="P0",
        parent_score=0.75,
        candidate_prompt="P1",
        candidate_score=0.75,
    )

    assert result.accepted is False
    assert result.selected_prompt == "P0"
    assert result.selected_source == "parent"


def test_selector_keeps_parent_when_candidate_is_worse() -> None:
    selector = PromptSelector()

    result = selector.select(
        parent_prompt="P0",
        parent_score=0.75,
        candidate_prompt="P1",
        candidate_score=0.6,
    )

    assert result.accepted is False
    assert result.selected_prompt == "P0"
    assert result.selected_source == "parent"
