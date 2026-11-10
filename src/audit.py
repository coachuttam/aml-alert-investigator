"""The audit log: one line per event, never edited, only appended to.

An event records what happened to an alert: a workflow step ran, or the analyst
approved or overrode the verdict. The original alert and the model's verdict are
never changed; the log is how anyone can see what was done, when and why.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
AUDIT_FOLDER = ROOT / "audit"          # local only, git-ignored
AUDIT_LOG = AUDIT_FOLDER / "audit_log.jsonl"


@dataclass(frozen=True)
class AuditEvent:
    """One thing that happened. 'before' and 'after' are short plain-text states."""
    time: str
    alert_id: str
    action: str
    before: str
    after: str
    note: str


def now_text() -> str:
    """The current time to the second, e.g. '2026-11-10T18:50:12'."""
    return datetime.now().isoformat(timespec="seconds")


def append_event(alert_id: str, action: str, before: str = "", after: str = "", note: str = "",
                 log_path: Path = AUDIT_LOG) -> AuditEvent:
    """Add one event to the end of the log and return it."""
    event = AuditEvent(now_text(), alert_id, action, before, after, note.strip())
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    return event


def read_events(log_path: Path = AUDIT_LOG) -> List[AuditEvent]:
    """Every event, oldest first. An empty list when nothing has happened yet."""
    if not log_path.exists():
        return []
    lines = log_path.read_text(encoding="utf-8").splitlines()
    return [AuditEvent(**json.loads(line)) for line in lines if line.strip()]
