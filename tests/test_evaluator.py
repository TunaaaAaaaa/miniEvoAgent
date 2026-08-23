from evo_repro import accuracy, evaluate_qa, exact_match, f1_score, normalize_answer


def test_normalize_answer_lowercases_removes_articles_punctuation_and_extra_spaces() -> None:
    assert normalize_answer("  The, Eiffel   Tower! ") == "eiffel tower"
    assert normalize_answer("An answer; with A title.") == "answer with title"


def test_exact_match_uses_normalized_answers() -> None:
    assert exact_match("The Eiffel Tower.", "eiffel tower") == 1.0
    assert exact_match("Paris", "London") == 0.0


def test_f1_score_counts_shared_tokens() -> None:
    assert f1_score("Eiffel Tower Paris", "Eiffel Tower") == 0.8
    assert f1_score("London", "Eiffel Tower") == 0.0


def test_f1_score_handles_yes_no_noanswer_special_cases() -> None:
    assert f1_score("yes", "yes") == 1.0
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes because evidence says so", "yes") == 0.0


def test_accuracy_checks_whether_label_appears_as_token_span() -> None:
    assert accuracy("The answer is the Eiffel Tower in Paris.", "Eiffel Tower") == 1.0
    assert accuracy("The answer is an Eiffel style tower.", "Eiffel Tower") == 0.0


def test_evaluate_qa_returns_expected_metric_dict() -> None:
    metrics = evaluate_qa("The Eiffel Tower.", "eiffel tower")

    assert metrics == {"em": 1.0, "f1": 1.0, "acc": 1.0}
