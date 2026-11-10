"""The fixed six-step investigation workflow, with guardrails.

    1 Load the alert           4 Classify (saved model answer, checked)
    2 Fetch the transactions   5 Draft the case note
    3 Compute the findings     6 Analyst decision (the only step a person does)

The steps always run in this order. No model plans anything; a small model calls
tools unreliably, so the order is fixed (see PRD.md). Steps 1 to 5 only read and
write local files. Nothing is filed, frozen or sent, and nothing can be exported,
before the analyst records a decision in step 6.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from src.audit import AUDIT_LOG, append_event, now_text
from src.case_note import GuardrailError, assert_clean_wording, build_findings, draft_case_note
from src.case_store import (AWAITING_ANALYST, CASE_FOLDER, DECIDED, NEEDS_REVIEW_STATUS, Case,
                            StepRecord, save_case)
from src.review import NEEDS_REVIEW, ReviewAlert, agreement_label, words_for
from src.schema import ALL_LABELS, RISK_LEVELS

ACTIONS = ("approve", "override")
PAST_TENSE = {"approve": "approved", "override": "overrode"}
WITHHELD_NOTE = "Note withheld: the wording check failed. Review by hand."


class DecisionError(Exception):
    """Raised when an analyst decision is incomplete or not allowed."""


class ExportBlockedError(Exception):
    """Raised when someone tries to export a case before the analyst has decided."""


def load_alert_step(alert: ReviewAlert) -> StepRecord:
    """Step 1: say which alert is being investigated."""
    return StepRecord(1, "Load alert", "done", f"Loaded {alert.alert_id} for {alert.customer_id}.",
                      [f"Set: {alert.split}", f"Window ends: {alert.window_end or 'not available'}"])


def fetch_transactions_step(transactions: List[dict]) -> StepRecord:
    """Step 2: count the transactions, or say plainly that they are not available."""
    if not transactions:
        return StepRecord(2, "Fetch transactions", "unavailable",
                          "Transactions are not available here.",
                          ["The raw file is made by `python -m src.generator`."])
    days = sorted(txn["day"] for txn in transactions)
    return StepRecord(2, "Fetch transactions", "done", f"Fetched {len(transactions)} transactions.",
                      [f"From {days[0]} to {days[-1]}."])


def compute_findings_step(alert: ReviewAlert) -> StepRecord:
    """Step 3: velocity and profile comparison, computed by Python from the features."""
    findings = build_findings(alert.features)
    return StepRecord(3, "Compute findings", "done", f"Computed {len(findings)} findings.", findings)


def classify_step(alert: ReviewAlert) -> StepRecord:
    """Step 4: show the model's saved answer. An answer that failed the checks goes to review."""
    if alert.model_typology == NEEDS_REVIEW:
        return StepRecord(4, "Classify", "needs_review", "The model's answer could not be validated.",
                          alert.model_problems)
    details = [f"Reason: {alert.model_reason}", agreement_label(alert),
               f"Tuned rules said: {words_for(alert.rules_tuned)} (rule {alert.rules_tuned_rule_id})."]
    return StepRecord(4, "Classify", "done",
                      f"{words_for(alert.model_typology)}, risk {alert.model_risk}.", details)


def draft_note_step(alert: ReviewAlert, findings: List[str]) -> tuple:
    """Step 5: returns (step record, note text). A failed wording check withholds the note."""
    try:
        note = draft_case_note(alert, findings)
    except GuardrailError as problem:
        return StepRecord(5, "Draft case note", "needs_review", str(problem), []), WITHHELD_NOTE
    return StepRecord(5, "Draft case note", "done", "Drafted the case note from the computed findings.", []), note


def waiting_step() -> StepRecord:
    """Step 6 before the analyst has decided."""
    return StepRecord(6, "Analyst decision", "waiting",
                      "Waiting for the analyst. Nothing has been filed or sent.", [])


def run_investigation(alert: ReviewAlert, transactions: List[dict], case_folder: Path = CASE_FOLDER,
                      log_path: Path = AUDIT_LOG) -> Case:
    """Run steps 1 to 5, log each one, save the case file and wait for the analyst."""
    steps = [load_alert_step(alert), fetch_transactions_step(transactions),
             compute_findings_step(alert), classify_step(alert)]
    note_step, note = draft_note_step(alert, steps[2].details)
    steps += [note_step, waiting_step()]
    for step in steps[:5]:
        append_event(alert.alert_id, f"step {step.number}: {step.name.lower()}", after=f"{step.status}: {step.summary}",
                     log_path=log_path)
    needs_review = any(step.status == "needs_review" for step in steps)
    case = Case(alert.alert_id, NEEDS_REVIEW_STATUS if needs_review else AWAITING_ANALYST, steps, note)
    save_case(case, case_folder)
    return case


def check_decision(action: str, typology: str, risk: str, note: str) -> None:
    """Refuse a decision that is incomplete, uses unknown labels, or uses accusing words."""
    if action not in ACTIONS:
        raise DecisionError("The decision must be approve or override.")
    if typology not in ALL_LABELS or risk not in RISK_LEVELS:
        raise DecisionError("Choose a typology and a risk level from the lists.")
    if action == "override" and not note.strip():
        raise DecisionError("Add a note to explain the override.")
    try:
        assert_clean_wording(note)
    except GuardrailError as problem:
        raise DecisionError(str(problem))


def record_decision(alert: ReviewAlert, case: Case, action: str, typology: str, risk: str, note: str,
                    case_folder: Path = CASE_FOLDER, log_path: Path = AUDIT_LOG) -> Case:
    """Step 6: save the analyst's decision beside the model's verdict, which stays unchanged."""
    check_decision(action, typology, risk, note)
    case.decision = {"time": now_text(), "action": action, "typology": typology, "risk": risk,
                     "note": note.strip()}
    case.status = DECIDED
    case.steps[5] = StepRecord(6, "Analyst decision", "done",
                               f"Analyst {PAST_TENSE[action]}: {words_for(typology)}, risk {risk}.", [note.strip()] if note.strip() else [])
    save_case(case, case_folder)
    append_event(alert.alert_id, f"decision: {action}", before=f"{words_for(alert.model_typology)} / {alert.model_risk}",
                 after=f"{words_for(typology)} / {risk}", note=note, log_path=log_path)
    return case


def decision_section(case: Case) -> List[str]:
    """The decision as Markdown lines for the exported note."""
    decision = case.decision
    lines = ["", "## Analyst decision",
             f"**{decision['action'].capitalize()}**: {words_for(decision['typology'])}, risk {decision['risk']} "
             f"({decision['time']})"]
    if decision["note"]:
        lines.append(f"Note: {decision['note']}")
    return lines


def export_markdown(case: Case) -> str:
    """The case note plus the analyst's decision. Blocked until the analyst has decided."""
    if case.decision is None:
        raise ExportBlockedError("Nothing can be exported before the analyst records a decision.")
    return "\n".join([case.note] + decision_section(case))


def export_json(case: Case) -> str:
    """The whole case file as JSON. Blocked until the analyst has decided."""
    if case.decision is None:
        raise ExportBlockedError("Nothing can be exported before the analyst records a decision.")
    return json.dumps(asdict(case), ensure_ascii=False, indent=2)
