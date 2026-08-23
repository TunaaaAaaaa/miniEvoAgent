from collections import Counter
import re
import string


def normalize_answer(text: str) -> str:
    """Normalize a QA answer using the common SQuAD/HotPotQA rules."""

    def lower(value: str) -> str:
        return value.lower()

    def remove_punctuation(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def fix_whitespace(value: str) -> str:
        return " ".join(value.split())

    return fix_whitespace(remove_articles(remove_punctuation(lower(text))))


def exact_match(prediction: str, label: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(label))


def f1_score(prediction: str, label: str) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_label = normalize_answer(label)

    if normalized_prediction in {"yes", "no", "noanswer"} and normalized_prediction != normalized_label:
        return 0.0
    if normalized_label in {"yes", "no", "noanswer"} and normalized_prediction != normalized_label:
        return 0.0

    prediction_tokens = normalized_prediction.split()
    label_tokens = normalized_label.split()
    common = Counter(prediction_tokens) & Counter(label_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(label_tokens)
    return 2 * precision * recall / (precision + recall)


def accuracy(prediction: str, label: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    label_tokens = normalize_answer(label).split()

    if not prediction_tokens or not label_tokens:
        return exact_match(prediction, label)

    for start in range(len(prediction_tokens) - len(label_tokens) + 1):
        if prediction_tokens[start : start + len(label_tokens)] == label_tokens:
            return 1.0
    return 0.0


def evaluate_qa(prediction: str, label: str) -> dict[str, float]:
    return {
        "em": exact_match(prediction, label),
        "f1": f1_score(prediction, label),
        "acc": accuracy(prediction, label),
    }
