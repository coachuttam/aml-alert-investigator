"""Scoring: precision, recall, F1 and false-positive rates.

All functions take two plain lists: the true labels and the predicted labels.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.schema import ALL_LABELS, NO_CONCERN


@dataclass(frozen=True)
class ClassScore:
    """How well one label was found."""
    precision: float
    recall: float
    f1: float


def safe_divide(top: float, bottom: float) -> float:
    """top / bottom, or 0.0 when there is nothing to divide by."""
    return top / bottom if bottom else 0.0


def score_label(true_labels: List[str], predicted: List[str], label: str) -> ClassScore:
    """Precision, recall and F1 for a single label."""
    pairs = list(zip(true_labels, predicted))
    true_positive = sum(1 for t, p in pairs if t == label and p == label)
    false_positive = sum(1 for t, p in pairs if t != label and p == label)
    false_negative = sum(1 for t, p in pairs if t == label and p != label)
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return ClassScore(precision, recall, f1)


def score_all_labels(true_labels: List[str], predicted: List[str]) -> Dict[str, ClassScore]:
    """A ClassScore for every label."""
    return {label: score_label(true_labels, predicted, label) for label in ALL_LABELS}


def macro_f1(true_labels: List[str], predicted: List[str]) -> float:
    """Average F1 over all labels, so rare labels count as much as common ones."""
    scores = score_all_labels(true_labels, predicted)
    return sum(score.f1 for score in scores.values()) / len(scores)


def false_positive_rates(
    true_labels: List[str], predicted: List[str], kinds: List[Optional[str]]
) -> Dict[str, float]:
    """Share of NO_CONCERN alerts wrongly flagged, split by kind of alert.

    `kinds` holds the hard-negative kind of each alert, or None for ordinary ones.
    The key "ordinary" covers the None group; "all_no_concern" covers everything.
    """
    flagged: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    for true, guess, kind in zip(true_labels, predicted, kinds):
        if true != NO_CONCERN:
            continue
        for group in ("all_no_concern", kind or "ordinary"):
            total[group] += 1
            if guess != NO_CONCERN:
                flagged[group] += 1
    return {group: safe_divide(flagged[group], total[group]) for group in total}
