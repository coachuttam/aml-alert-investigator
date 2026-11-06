"""System A: a threshold-rules baseline (no learning involved).

Each rule looks at one computed number. Rules run in a fixed order and the first
one that fires decides the label. The rules never output a risk level.

Two settings are compared:
  DEFAULT_CONFIG  thresholds written down before looking at any results.
  tune_on_training_set()  the best thresholds found on the training set only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Dict, List, Tuple

from src.metrics import macro_f1
from src.schema import NO_CONCERN


@dataclass(frozen=True)
class RuleConfig:
    """The thresholds. Each one is "fire when the number is at least this"."""
    just_under_min: int = 3          # cash deposits just under the reporting limit
    sender_min: int = 5              # different senders in the busiest week
    cash_out_min: int = 50           # percent of that burst sent on within days
    return_min: int = 2              # transfers that came back
    large_min: int = 1               # credits of Rs 50,000 or more
    quick_exit_min: int = 70         # percent of large credits sent on within 2 days


DEFAULT_CONFIG = RuleConfig()

# Values tried when tuning. Small grids keep the search fast and easy to read.
GRID = {
    "just_under_min": (3, 4, 5, 6, 8),
    "sender_min": (5, 8, 10, 12),
    "cash_out_min": (30, 50, 70, 85),
    "return_min": (1, 2, 3),
    "large_min": (1, 2),
    "quick_exit_min": (50, 70, 80, 90),
}


def classify(features: dict, config: RuleConfig = DEFAULT_CONFIG) -> Tuple[str, str]:
    """Return (typology, rule_id). Order: R1 structuring, R2 mule, R3 round trip, R4 pass-through."""
    if features["just_under_count"] >= config.just_under_min:
        return "STRUCTURING", "R1"
    if features["sender_count"] >= config.sender_min and features["cash_out_percent"] >= config.cash_out_min:
        return "MULE_ACCOUNT", "R2"
    if features["return_count"] >= config.return_min:
        return "ROUND_TRIPPING", "R3"
    if features["large_count"] >= config.large_min and features["quick_exit_percent"] >= config.quick_exit_min:
        return "RAPID_PASS_THROUGH", "R4"
    return NO_CONCERN, "R0"


def predict_all(examples: List[dict], config: RuleConfig = DEFAULT_CONFIG) -> List[str]:
    """The rules' label for every example."""
    return [classify(example["features"], config)[0] for example in examples]


def all_configs() -> List[RuleConfig]:
    """Every combination of the grid values."""
    names = list(GRID)
    return [replace(DEFAULT_CONFIG, **dict(zip(names, values)))
            for values in product(*(GRID[name] for name in names))]


def tune_on_training_set(train_examples: List[dict]) -> RuleConfig:
    """Pick the thresholds with the best macro-F1 on the training set. Never sees held-out."""
    true_labels = [example["label"]["typology"] for example in train_examples]
    best_config, best_score = DEFAULT_CONFIG, -1.0
    for config in all_configs():
        score = macro_f1(true_labels, predict_all(train_examples, config))
        if score > best_score:
            best_config, best_score = config, score
    return best_config
