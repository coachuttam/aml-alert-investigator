"""Turns raw alerts into training examples: summary text plus the true label.

Run it to write data/train.jsonl, validation.jsonl, test.jsonl and heldout.jsonl:
    python -m src.dataset
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from src.features import compute_features, reason_for, risk_level
from src.generator import generate_all
from src.models import RawAlert
from src.schema import SPLITS
from src.templates import pick_template, render_summary


def build_example(raw: RawAlert) -> dict:
    """One training example: what the model reads, and the answer it should give."""
    features = compute_features(raw)
    template_id = pick_template(raw.alert_id, raw.template_group)
    feature_values = asdict(features)
    # Store payees as lists, which is what a JSON file gives back.
    feature_values["top_payees"] = [list(payee) for payee in features.top_payees]
    return {
        "alert_id": raw.alert_id,
        "customer_id": raw.customer.customer_id,
        "split": raw.split,
        "template_id": template_id,
        "hard_negative_kind": raw.hard_negative_kind,
        "summary": render_summary(template_id, features),
        "label": {
            "typology": raw.typology,
            "risk": risk_level(raw.typology, features),
            "reason": reason_for(raw, features),
        },
        "features": feature_values,
    }


def build_dataset(alerts_by_split: Dict[str, List[RawAlert]]) -> Dict[str, List[dict]]:
    """Build the examples for every split."""
    return {split: [build_example(raw) for raw in alerts_by_split[split]] for split in SPLITS}


def save_dataset(dataset: Dict[str, List[dict]], folder: Path) -> None:
    """Write one .jsonl file per split into `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    for split, examples in dataset.items():
        lines = [json.dumps(example, sort_keys=True, ensure_ascii=False) for example in examples]
        (folder / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_split(path: Path) -> List[dict]:
    """Read one split written by save_dataset."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def print_training_summary(examples: List[dict]) -> None:
    """Show the mix of typologies and risk levels in the training set."""
    mix = Counter((e["label"]["typology"], e["label"]["risk"]) for e in examples)
    print(f"{'typology':<20}{'low':>6}{'medium':>8}{'high':>6}")
    for typology in sorted({typology for typology, _ in mix}):
        counts = [mix[(typology, risk)] for risk in ("low", "medium", "high")]
        print(f"{typology:<20}{counts[0]:>6}{counts[1]:>8}{counts[2]:>6}")


def main() -> None:
    """Generate the alerts, build the examples and write data/*.jsonl."""
    folder = Path(__file__).resolve().parent.parent / "data"
    dataset = build_dataset(generate_all())
    save_dataset(dataset, folder)
    for split, examples in dataset.items():
        print(f"{split:<11} {len(examples):>5} examples")
    print()
    print_training_summary(dataset["train"])
    print(f"\nWritten to {folder}")


if __name__ == "__main__":
    main()
