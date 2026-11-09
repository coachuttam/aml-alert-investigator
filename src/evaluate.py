"""Scores the systems and prints a results table.

    python -m src.evaluate            validation and test only
    python -m src.evaluate --final    also scores the held-out set (do this once, at the end)

Rules are scored directly. Model answers are read from results/<system>_<split>.jsonl,
which src/predict.py writes on a GPU machine. Systems with no file yet are skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.dataset import load_split
from src.metrics import false_positive_rates, macro_f1, score_all_labels
from src.rules import DEFAULT_CONFIG, RuleConfig, predict_all, tune_on_training_set
from src.schema import ALL_LABELS

ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = ROOT / "data"
RESULTS_FOLDER = ROOT / "results"
DEFAULT_SPLITS = ("validation", "test")
MODEL_SYSTEMS = {"B base model": "base", "C fine-tuned": "finetuned"}
INVALID = "INVALID"          # label used when a model gives no usable typology
UNFAITHFUL_PROBLEM = "reason uses a number that is not in the summary"


def evaluate_predictions(examples: List[dict], predicted: List[str]) -> dict:
    """Macro-F1, per-label scores and false-positive rates for one system on one split."""
    true_labels = [example["label"]["typology"] for example in examples]
    kinds = [example["hard_negative_kind"] for example in examples]
    return {
        "macro_f1": macro_f1(true_labels, predicted),
        "per_label": score_all_labels(true_labels, predicted),
        "false_positive_rates": false_positive_rates(true_labels, predicted, kinds),
    }


def load_predictions(path: Path) -> Dict[str, dict]:
    """Saved model answers, keyed by alert id."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {row["alert_id"]: row for row in rows}


def typology_of(row: Optional[dict]) -> str:
    """The typology a model gave, or INVALID when it gave none usable."""
    if row and row.get("typology") in ALL_LABELS:
        return row["typology"]
    return INVALID


def share(count: int, total: int) -> float:
    """count / total, or 0.0 for an empty set."""
    return count / total if total else 0.0


def model_extras(examples: List[dict], answers: Dict[str, dict]) -> dict:
    """Checks only models need: valid answers, faithful reasons, correct risk level."""
    rows = [answers.get(example["alert_id"]) for example in examples]
    valid = sum(1 for row in rows if row and row["valid"])
    unfaithful = sum(1 for row in rows if row and UNFAITHFUL_PROBLEM in row["problems"])
    right_risk = sum(1 for example, row in zip(examples, rows)
                     if row and row.get("risk") == example["label"]["risk"])
    return {
        "valid_answers": share(valid, len(examples)),
        "faithful_reasons": 1 - share(unfaithful, len(examples)),
        "risk_accuracy": share(right_risk, len(examples)),
    }


def print_report(name: str, split: str, result: dict) -> None:
    """Print one system's scores on one split."""
    print(f"\n{name} on {split}: macro-F1 {result['macro_f1']:.3f}")
    for label, score in result["per_label"].items():
        print(f"  {label:<20} precision {score.precision:.2f}  recall {score.recall:.2f}  F1 {score.f1:.2f}")
    print("  false-positive rate on NO_CONCERN alerts:")
    for group, rate in sorted(result["false_positive_rates"].items()):
        print(f"    {group:<28}{rate:6.1%}")
    extras = result.get("extras")
    if extras:
        print(f"  valid answers {extras['valid_answers']:.1%}   faithful reasons "
              f"{extras['faithful_reasons']:.1%}   risk accuracy {extras['risk_accuracy']:.1%}")


def describe(config: RuleConfig) -> str:
    """The thresholds as one line."""
    return ", ".join(f"{name}={value}" for name, value in vars(config).items())


def score_rules(splits: List[str]) -> None:
    """Score both rules baselines on the chosen splits."""
    train = load_split(DATA_FOLDER / "train.jsonl")
    systems = {"A1 rules (default)": DEFAULT_CONFIG,
               "A2 rules (tuned on train)": tune_on_training_set(train)}
    for name, config in systems.items():
        print(f"\n{name}: {describe(config)}")
        for split in splits:
            examples = load_split(DATA_FOLDER / f"{split}.jsonl")
            print_report(name, split, evaluate_predictions(examples, predict_all(examples, config)))


def score_models(splits: List[str], results_folder: Path = RESULTS_FOLDER) -> None:
    """Score every model system that has a saved answers file."""
    for name, file_prefix in MODEL_SYSTEMS.items():
        for split in splits:
            path = results_folder / f"{file_prefix}_{split}.jsonl"
            if not path.exists():
                print(f"\n{name} on {split}: no answers yet ({path.name})")
                continue
            examples = load_split(DATA_FOLDER / f"{split}.jsonl")
            answers = load_predictions(path)
            predicted = [typology_of(answers.get(e["alert_id"])) for e in examples]
            result = evaluate_predictions(examples, predicted)
            result["extras"] = model_extras(examples, answers)
            print_report(name, split, result)


def main() -> None:
    """Score the rules baselines and any saved model answers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="also score the held-out set")
    splits = list(DEFAULT_SPLITS) + (["heldout"] if parser.parse_args().final else [])
    score_rules(splits)
    score_models(splits)


if __name__ == "__main__":
    main()
