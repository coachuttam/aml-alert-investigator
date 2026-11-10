"""The numbers for the Evaluation tab: one row per system, test and held-out side by side.

It reuses the scoring code in evaluate.py, so the screen can never disagree with
the README table.
"""
from __future__ import annotations

from typing import Dict, List

from src.dataset import load_split
from src.evaluate import (evaluate_predictions, load_predictions, model_extras, typology_of)
from src.review import DATA_FOLDER, RESULTS_FOLDER
from src.rules import DEFAULT_CONFIG, RuleConfig, predict_all

SCORE_SPLITS = ("test", "heldout")
MODEL_FILES = {"B base model": "base", "C fine-tuned": "finetuned"}


def score_one_system(examples: List[dict], predicted: List[str]) -> dict:
    """Macro-F1 and the false-alarm rate on normal alerts for one system on one split."""
    result = evaluate_predictions(examples, predicted)
    return {"macro_f1": result["macro_f1"],
            "false_alarms": result["false_positive_rates"]["all_no_concern"]}


def rules_scores(config: RuleConfig, examples_by_split: Dict[str, List[dict]]) -> dict:
    """One rules system scored on every split."""
    return {split: score_one_system(examples, predict_all(examples, config))
            for split, examples in examples_by_split.items()}


def model_scores(file_prefix: str, examples_by_split: Dict[str, List[dict]]) -> dict:
    """One model system scored on every split that has a saved answers file."""
    scores = {}
    for split, examples in examples_by_split.items():
        path = RESULTS_FOLDER / f"{file_prefix}_{split}.jsonl"
        if not path.exists():
            continue
        answers = load_predictions(path)
        predicted = [typology_of(answers.get(example["alert_id"])) for example in examples]
        scores[split] = score_one_system(examples, predicted)
        scores[split]["valid_answers"] = model_extras(examples, answers)["valid_answers"]
    return scores


def build_score_table(tuned_config: RuleConfig) -> Dict[str, dict]:
    """System name -> split -> scores. Held-out rows are read from files already saved."""
    examples_by_split = {split: load_split(DATA_FOLDER / f"{split}.jsonl") for split in SCORE_SPLITS}
    table = {
        "A1 rules (default)": rules_scores(DEFAULT_CONFIG, examples_by_split),
        "A2 rules (tuned on train)": rules_scores(tuned_config, examples_by_split),
    }
    for name, prefix in MODEL_FILES.items():
        table[name] = model_scores(prefix, examples_by_split)
    return table
