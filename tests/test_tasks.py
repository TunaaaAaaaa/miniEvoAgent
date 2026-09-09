import math

import pytest

from evo_repro import GSM8KTask, NQTask, QATask


def test_nq_scores_best_alias_and_keeps_punctuation_for_containment():
    task = NQTask()
    assert task.evaluate("The UK", ["United Kingdom", "UK"]) == {
        "em": 1.0, "f1": 1.0, "acc": 1.0,
    }
    assert task.evaluate("New York City", ["New York", "NYC"])["f1"] == pytest.approx(0.8)
    assert task.evaluate("artist", ["art"])["acc"] == 0.0
    assert task.evaluate("New-York", ["New York"])["acc"] == 0.0
    assert task.evaluate("A café!", ["cafe\u0301"])["acc"] == 1.0


@pytest.mark.parametrize("prediction,label,score", [
    ("We get 6*7=42. Final: 42", "reasoning\n#### 42", 1.0),
    ("$1,200", "#### 1200", 1.0),
    ("-3.5", "#### -3.50", 1.0),
    ("42 then 43", "#### 42", 0.0),
    ("no answer", "#### 42", 0.0),
    ("0.0000001", "#### 0", 1.0),
    ("9" * 400, "#### 42", 0.0),
])
def test_gsm8k_numeric_solve_rate(prediction, label, score):
    assert GSM8KTask().evaluate(prediction, label) == {"solve_rate": score}


@pytest.mark.parametrize("task,prediction,label", [
    (QATask(), "answer", ["answer"]),
    (NQTask(), "answer", []),
    (NQTask(), "answer", "answer"),
    (GSM8KTask(), "42", "no numeric label"),
])
def test_invalid_labels_fail_explicitly(task, prediction, label):
    with pytest.raises(ValueError):
        task.evaluate(prediction, label)


def test_task_aggregation_rejects_missing_nonfinite_or_empty_scores():
    task = GSM8KTask()
    with pytest.raises(KeyError):
        task.aggregate([{"f1": 1.0}])
    with pytest.raises(ValueError, match="finite"):
        task.aggregate([{"solve_rate": math.nan}])
    with pytest.raises(ValueError, match="empty"):
        task.aggregate([])
    assert task.aggregate([{"solve_rate": 0.0}, {"solve_rate": 1.0}]) == 0.5
