"""The data behind the review screen. No Streamlit in here, so it is easy to test.

For each alert it gathers: the summary the model read, the fine-tuned model's
verdict, what the rules and the base model said, and the true label.
Analyst decisions live in the case files and the audit log (see investigator.py).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.dataset import load_split
from src.evaluate import load_predictions, typology_of
from src.features import LARGE_AMOUNT
from src.patterns import CASH_REPORTING_LIMIT, JUST_UNDER_HIGH, JUST_UNDER_LOW
from src.rules import DEFAULT_CONFIG, classify, tune_on_training_set
from src.schema import NO_CONCERN

ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = ROOT / "data"
RAW_FOLDER = DATA_FOLDER / "raw"
RESULTS_FOLDER = ROOT / "results"

RISK_ORDER = {"high": 0, "medium": 1, "low": 2, "unknown": 3}
NEEDS_REVIEW = "NEEDS_REVIEW"     # shown when the model's answer could not be validated

# Plain words for the typology names (see DESIGN_SYSTEM.md).
TYPOLOGY_WORDS = {
    "STRUCTURING": "Structuring",
    "RAPID_PASS_THROUGH": "Rapid pass-through",
    "MULE_ACCOUNT": "Mule account",
    "ROUND_TRIPPING": "Round-tripping",
    NO_CONCERN: "No concern",
    NEEDS_REVIEW: "Needs review",
}

HELDOUT_NOTE = "held-out (run once)"
STREAMLIT_CLOUD_FOLDER = Path("/mount/src")       # where Streamlit Community Cloud puts the repo


@dataclass(frozen=True)
class ReviewAlert:
    """Everything the screen shows about one alert."""
    alert_id: str
    split: str
    customer_id: str
    summary: str
    features: dict
    true_typology: str
    true_risk: str
    look_alike_kind: Optional[str]
    model_typology: str          # NEEDS_REVIEW when the answer was not valid
    model_risk: str
    model_reason: str
    model_problems: List[str]
    rules_default: str
    rules_tuned: str
    rules_tuned_rule_id: str
    base_typology: str
    window_end: str


def is_shared_demo(environment=None, cloud_folder: Path = STREAMLIT_CLOUD_FOLDER) -> bool:
    """True when the app runs on a public host (a Hugging Face Space or Streamlit Community Cloud).

    There, every visitor shares the same saved decisions, so the app shows a notice.
    """
    environment = os.environ if environment is None else environment
    return bool(environment.get("SPACE_ID")) or cloud_folder.exists()


def words_for(typology: str) -> str:
    """'RAPID_PASS_THROUGH' becomes 'Rapid pass-through'."""
    return TYPOLOGY_WORDS.get(typology, typology)


def tuned_rules_config():
    """The thresholds tuned on the training set only (takes about 5 seconds)."""
    return tune_on_training_set(load_split(DATA_FOLDER / "train.jsonl"))


def read_answers(system_prefix: str, split: str) -> Dict[str, dict]:
    """Saved answers for one system and split, or an empty dict if the file is missing."""
    path = RESULTS_FOLDER / f"{system_prefix}_{split}.jsonl"
    return load_predictions(path) if path.exists() else {}


def read_window_ends(split: str) -> Dict[str, str]:
    """Window end date per alert from the raw file, if the raw file exists."""
    path = RAW_FOLDER / f"{split}.jsonl"
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {row["alert_id"]: row["window_end"] for row in rows}


def read_transactions(split: str) -> Dict[str, List[dict]]:
    """Transactions per alert from the raw file, or an empty dict if the raw file is missing."""
    path = RAW_FOLDER / f"{split}.jsonl"
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return {row["alert_id"]: row["transactions"] for row in rows}


def build_review_alert(example: dict, tuned_config, fine_tuned: Optional[dict],
                       base: Optional[dict], window_end: str) -> ReviewAlert:
    """Join one dataset example with the rules' labels and the saved model answers."""
    default_label, _ = classify(example["features"], DEFAULT_CONFIG)
    tuned_label, tuned_rule = classify(example["features"], tuned_config)
    usable = bool(fine_tuned and fine_tuned.get("valid"))
    return ReviewAlert(
        alert_id=example["alert_id"],
        split=example["split"],
        customer_id=example["customer_id"],
        summary=example["summary"],
        features=example["features"],
        true_typology=example["label"]["typology"],
        true_risk=example["label"]["risk"],
        look_alike_kind=example["hard_negative_kind"],
        model_typology=fine_tuned["typology"] if usable else NEEDS_REVIEW,
        model_risk=fine_tuned["risk"] if usable else "unknown",
        model_reason=(fine_tuned.get("reason") or "") if usable else "",
        model_problems=list(fine_tuned["problems"]) if fine_tuned else ["no saved answer"],
        rules_default=default_label,
        rules_tuned=tuned_label,
        rules_tuned_rule_id=tuned_rule,
        base_typology=typology_of(base),
        window_end=window_end,
    )


def sort_for_queue(alerts: List[ReviewAlert]) -> List[ReviewAlert]:
    """Highest risk first, then the most recent window first."""
    by_date = sorted(alerts, key=lambda alert: alert.window_end, reverse=True)
    return sorted(by_date, key=lambda alert: RISK_ORDER.get(alert.model_risk, 3))


def load_review_alerts(split: str, tuned_config) -> List[ReviewAlert]:
    """All alerts of one split, ready to show, in queue order."""
    examples = load_split(DATA_FOLDER / f"{split}.jsonl")
    fine_tuned = read_answers("finetuned", split)
    base = read_answers("base", split)
    window_ends = read_window_ends(split)
    alerts = [build_review_alert(example, tuned_config, fine_tuned.get(example["alert_id"]),
                                 base.get(example["alert_id"]), window_ends.get(example["alert_id"], ""))
              for example in examples]
    return sort_for_queue(alerts)


def supports_verdict(transaction: dict, typology: str) -> bool:
    """True for rows we can point to as evidence for structuring or pass-through verdicts.

    Mule and round-trip evidence needs several rows together, so those rows are not marked.
    """
    if transaction["direction"] != "credit":
        return False
    if typology == "STRUCTURING":
        just_under = CASH_REPORTING_LIMIT * JUST_UNDER_LOW <= transaction["amount"] <= CASH_REPORTING_LIMIT * JUST_UNDER_HIGH
        return transaction["channel"] == "cash" and just_under
    if typology == "RAPID_PASS_THROUGH":
        return transaction["amount"] >= LARGE_AMOUNT
    return False


def agreement_label(alert: ReviewAlert) -> str:
    """One plain line on whether the model and the tuned rules agree."""
    if alert.model_typology == NEEDS_REVIEW:
        return "Model answer could not be validated: review by hand."
    if alert.model_typology == alert.rules_tuned:
        return "The model and the tuned rules agree."
    return "The model and the tuned rules disagree: look closely."
