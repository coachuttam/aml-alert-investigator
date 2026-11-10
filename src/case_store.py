"""Case memory: one JSON case file per alert that the workflow steps add to.

Reopening an alert loads its case file, so the steps, the draft note and the
analyst's decision are still there. Case files are local and git-ignored.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from src.audit import AUDIT_FOLDER

CASE_FOLDER = AUDIT_FOLDER / "cases"

# Where a case stands.
AWAITING_ANALYST = "awaiting_analyst"
NEEDS_REVIEW_STATUS = "needs_review"      # the model's answer could not be validated
DECIDED = "decided"


@dataclass
class StepRecord:
    """What one workflow step did. 'details' are short plain-English lines."""
    number: int
    name: str
    status: str                      # done, needs_review, unavailable or waiting
    summary: str
    details: List[str] = field(default_factory=list)


@dataclass
class Case:
    """Everything the workflow knows about one alert."""
    alert_id: str
    status: str
    steps: List[StepRecord]
    note: str
    decision: Optional[dict] = None   # set once, by the analyst: action, typology, risk, note, time


def case_path(alert_id: str, folder: Path = CASE_FOLDER) -> Path:
    return folder / f"{alert_id}.json"


def save_case(case: Case, folder: Path = CASE_FOLDER) -> None:
    """Write the case file, replacing the earlier version of the same case."""
    folder.mkdir(parents=True, exist_ok=True)
    case_path(case.alert_id, folder).write_text(
        json.dumps(asdict(case), ensure_ascii=False, indent=2), encoding="utf-8")


def load_case(alert_id: str, folder: Path = CASE_FOLDER) -> Optional[Case]:
    """The saved case, or None if this alert has not been investigated yet."""
    path = case_path(alert_id, folder)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = [StepRecord(**step) for step in data.pop("steps")]
    return Case(steps=steps, **data)


def count_cases(folder: Path = CASE_FOLDER) -> int:
    """How many alerts have a case file."""
    return len(list(folder.glob("*.json"))) if folder.exists() else 0
