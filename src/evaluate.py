"""Scores the systems and prints a results table.

    python -m src.evaluate            validation and test only
    python -m src.evaluate --final    also runs the held-out set (do this once, at the end)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from src.dataset import load_split
from src.metrics import false_positive_rates, macro_f1, score_all_labels
from src.rules import DEFAULT_CONFIG, RuleConfig, predict_all, tune_on_training_set

DATA_FOLDER = Path(__file__).resolve().parent.parent / "data"
DEFAULT_SPLITS = ("validation", "test")


def evaluate_predictions(examples: List[dict], predicted: List[str]) -> dict:
    """Macro-F1, per-label scores and false-positive rates for one system on one split."""
    true_labels = [example["label"]["typology"] for example in examples]
    kinds = [example["hard_negative_kind"] for example in examples]
    return {
        "macro_f1": macro_f1(true_labels, predicted),
        "per_label": score_all_labels(true_labels, predicted),
        "false_positive_rates": false_positive_rates(true_labels, predicted, kinds),
    }


def print_report(name: str, split: str, result: dict) -> None:
    """Print one system's scores on one split."""
    print(f"\n{name} on {split}: macro-F1 {result['macro_f1']:.3f}")
    for label, score in result["per_label"].items():
        print(f"  {label:<20} precision {score.precision:.2f}  recall {score.recall:.2f}  F1 {score.f1:.2f}")
    print("  false-positive rate on NO_CONCERN alerts:")
    for group, rate in sorted(result["false_positive_rates"].items()):
        print(f"    {group:<28}{rate:6.1%}")


def describe(config: RuleConfig) -> str:
    """The thresholds as one line."""
    return ", ".join(f"{name}={value}" for name, value in vars(config).items())


def main() -> None:
    """Score the rules baselines on the chosen splits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="also score the held-out set")
    splits = list(DEFAULT_SPLITS) + (["heldout"] if parser.parse_args().final else [])

    train = load_split(DATA_FOLDER / "train.jsonl")
    systems: Dict[str, RuleConfig] = {
        "A1 rules (default)": DEFAULT_CONFIG,
        "A2 rules (tuned on train)": tune_on_training_set(train),
    }
    for name, config in systems.items():
        print(f"\n{name}: {describe(config)}")
        for split in splits:
            examples = load_split(DATA_FOLDER / f"{split}.jsonl")
            print_report(name, split, evaluate_predictions(examples, predict_all(examples, config)))


if __name__ == "__main__":
    main()
