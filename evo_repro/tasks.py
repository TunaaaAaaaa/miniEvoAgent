"""Task-specific input mapping and scoring for the generic evolution loop."""

from abc import ABC, abstractmethod
from copy import deepcopy
import math
import re
from statistics import mean
from typing import Any
import unicodedata

from .base import BaseModel
from .evaluators import evaluate_qa
from .messages import Message


class TaskAdapter(BaseModel, ABC):
    name: str
    score_key: str
    input_key: str = "question"

    def collate(self, value: Any) -> dict[str, Any]:
        return {self.input_key: deepcopy(value)}

    def prediction(self, message: Message) -> Any:
        return message.content

    @abstractmethod
    def evaluate(self, prediction: Any, label: Any) -> dict[str, float]:
        raise NotImplementedError

    def aggregate(self, metrics: list[dict[str, float]]) -> float:
        if not metrics:
            raise ValueError("Cannot score an empty evaluation set.")
        values = [float(item[self.score_key]) for item in metrics]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Metric {self.score_key!r} must be finite.")
        return mean(values)


class QATask(TaskAdapter):
    """Backward-compatible, single-answer QA scoring."""

    name: str = "qa"
    score_key: str = "f1"

    def evaluate(self, prediction: Any, label: Any) -> dict[str, float]:
        if not isinstance(prediction, str) or not isinstance(label, str):
            raise ValueError("QA prediction and label must be strings.")
        return evaluate_qa(prediction, label)


def _answer_tokens(text: str) -> list[str]:
    # DPR-style Unicode letter/number/mark spans; punctuation stays a separate token.
    tokens: list[str] = []
    span = ""
    for char in unicodedata.normalize("NFD", text).lower():
        if unicodedata.category(char)[0] in "LNM":
            span += char
        else:
            if span:
                tokens.append(span)
                span = ""
            if not char.isspace() and unicodedata.category(char)[0] != "C":
                tokens.append(char)
    if span:
        tokens.append(span)
    return tokens


class NQTask(TaskAdapter):
    """NQ short-answer aliases: best EM/F1 and DPR-style answer containment."""

    name: str = "nq"
    score_key: str = "f1"

    def evaluate(self, prediction: Any, label: Any) -> dict[str, float]:
        if (not isinstance(prediction, str) or not isinstance(label, list)
                or not label or not all(isinstance(answer, str) for answer in label)):
            raise ValueError("NQ requires a text prediction and a nonempty list of answers.")
        scores = [evaluate_qa(prediction, answer) for answer in label]
        predicted = _answer_tokens(prediction)
        contains = False
        for answer in label:
            tokens = _answer_tokens(answer)
            if tokens and any(predicted[i:i + len(tokens)] == tokens
                              for i in range(len(predicted) - len(tokens) + 1)):
                contains = True
        return {"em": max(s["em"] for s in scores),
                "f1": max(s["f1"] for s in scores), "acc": float(contains)}


class GSM8KTask(TaskAdapter):
    """EvoAgentX-style last-number scoring, including worked-solution labels."""

    name: str = "gsm8k"
    score_key: str = "solve_rate"

    @staticmethod
    def extract_number(text: Any) -> float | None:
        numbers = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if not numbers:
            return None
        value = float(numbers[-1].replace(",", ""))
        return value if math.isfinite(value) else None

    def evaluate(self, prediction: Any, label: Any) -> dict[str, float]:
        expected = self.extract_number(label)
        if expected is None:
            raise ValueError("GSM8K label must contain a finite numeric answer.")
        actual = self.extract_number(prediction)
        return {"solve_rate": float(actual is not None and abs(actual - expected) < 1e-6)}
