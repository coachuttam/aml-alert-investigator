"""Tests for the six-step workflow, its guardrails and the case note."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.audit import read_events
from src.case_note import GuardrailError, assert_clean_wording, build_findings, draft_case_note
from src.case_store import AWAITING_ANALYST, DECIDED, NEEDS_REVIEW_STATUS, load_case
from src.dataset import load_split
from src.evidence import findings_for_rows
from src.investigator import (DecisionError, ExportBlockedError, export_json, export_markdown,
                              record_decision, run_investigation)
from src.review import ReviewAlert, build_review_alert
from src.rules import DEFAULT_CONFIG
from src.schema import numbers_in

DATA_FOLDER = Path(__file__).resolve().parent.parent / "data"
TRANSACTIONS = [{"txn_id": "A-1-T001", "day": "2026-06-01", "direction": "credit", "channel": "cash",
                 "amount": 45000, "counterparty": "Cash deposit"},
                {"txn_id": "A-1-T002", "day": "2026-06-03", "direction": "credit", "channel": "cash",
                 "amount": 46000, "counterparty": "Cash deposit"}]


def make_features() -> dict:
    return {"income": 30000, "credits_total": 91000, "credits_ratio": 3.0, "debits_total": 20000,
            "just_under_count": 2, "just_under_total": 91000, "just_under_span_days": 2,
            "large_count": 0, "large_total": 0, "quick_exit_percent": 0,
            "sender_count": 0, "sender_total": 0, "sender_span_days": 0, "cash_out_percent": 0,
            "return_count": 0, "return_total": 0, "cash_count": 2, "account_age_months": 40,
            "occupation": "shopkeeper"}


def make_alert(valid: bool = True, reason: str = "2 cash deposits just under the limit.") -> ReviewAlert:
    example = {"alert_id": "A-1", "customer_id": "C-1", "split": "test", "summary": "x",
               "features": make_features(), "hard_negative_kind": None, "template_id": "digest",
               "label": {"typology": "STRUCTURING", "risk": "medium", "reason": "x"}}
    answer = {"valid": valid, "typology": "STRUCTURING", "risk": "medium", "reason": reason,
              "problems": [] if valid else ["not valid JSON"]}
    return build_review_alert(example, DEFAULT_CONFIG, answer, None, "2026-06-05")


def run(tmp_path, alert=None, transactions=TRANSACTIONS):
    alert = alert or make_alert()
    case = run_investigation(alert, transactions, tmp_path / "cases", tmp_path / "log.jsonl")
    return alert, case


def test_the_workflow_runs_six_steps_in_a_fixed_order(tmp_path):
    _, case = run(tmp_path)
    assert [step.number for step in case.steps] == [1, 2, 3, 4, 5, 6]
    assert [step.status for step in case.steps] == ["done"] * 5 + ["waiting"]
    assert case.status == AWAITING_ANALYST
    assert case.decision is None


def test_steps_one_to_five_are_logged_and_the_case_file_is_saved(tmp_path):
    alert, _ = run(tmp_path)
    events = read_events(tmp_path / "log.jsonl")
    assert len(events) == 5
    assert load_case(alert.alert_id, tmp_path / "cases") is not None


def test_a_model_answer_that_failed_the_checks_sends_the_case_to_needs_review(tmp_path):
    _, case = run(tmp_path, make_alert(valid=False))
    assert case.status == NEEDS_REVIEW_STATUS
    assert case.steps[3].status == "needs_review"
    assert "could not be validated" in case.note


def test_missing_transactions_are_reported_not_hidden(tmp_path):
    _, case = run(tmp_path, transactions=[])
    assert case.steps[1].status == "unavailable"
    assert case.status == AWAITING_ANALYST


def test_the_case_note_is_refused_if_it_would_contain_an_accusing_word(tmp_path):
    alert = make_alert(reason="Looks like fraud to me.")
    with pytest.raises(GuardrailError):
        draft_case_note(alert, findings_for_rows(alert.features, TRANSACTIONS))
    _, case = run(tmp_path, alert)
    assert case.steps[4].status == "needs_review"
    assert "withheld" in case.note


def test_every_finding_uses_only_numbers_from_the_alert_summary():
    # The only number the findings add themselves is the 2-day quick-exit window.
    for split in ("validation", "test"):
        for example in load_split(DATA_FOLDER / f"{split}.jsonl"):
            findings = " ".join(build_findings(example["features"]))
            assert numbers_in(findings) - numbers_in(example["summary"]) <= {"2"}


def test_nothing_can_be_exported_before_the_analyst_decides(tmp_path):
    _, case = run(tmp_path)
    with pytest.raises(ExportBlockedError):
        export_markdown(case)
    with pytest.raises(ExportBlockedError):
        export_json(case)


def test_an_approval_is_saved_logged_and_exportable(tmp_path):
    alert, case = run(tmp_path)
    case = record_decision(alert, case, "approve", "STRUCTURING", "medium", "", tmp_path / "cases", tmp_path / "log.jsonl")
    assert case.status == DECIDED
    assert load_case(alert.alert_id, tmp_path / "cases").decision["action"] == "approve"
    last_event = read_events(tmp_path / "log.jsonl")[-1]
    assert last_event.action == "decision: approve"
    assert last_event.before == "Structuring / medium"
    assert "Analyst decision" in export_markdown(case)
    assert '"decision"' in export_json(case)


def test_an_override_keeps_the_model_verdict_and_records_the_analyst_one(tmp_path):
    alert, case = run(tmp_path)
    record_decision(alert, case, "override", "NO_CONCERN", "low", "Cash is normal for this shop.",
                    tmp_path / "cases", tmp_path / "log.jsonl")
    last_event = read_events(tmp_path / "log.jsonl")[-1]
    assert (last_event.before, last_event.after) == ("Structuring / medium", "No concern / low")
    assert alert.model_typology == "STRUCTURING"


@pytest.mark.parametrize("action, typology, risk, note", [
    ("override", "NO_CONCERN", "low", "   "),
    ("delete", "NO_CONCERN", "low", "note"),
    ("approve", "MADE_UP", "low", ""),
    ("approve", "STRUCTURING", "extreme", ""),
    ("override", "NO_CONCERN", "low", "This customer is guilty."),
])
def test_incomplete_or_accusing_decisions_are_refused(tmp_path, action, typology, risk, note):
    alert, case = run(tmp_path)
    with pytest.raises(DecisionError):
        record_decision(alert, case, action, typology, risk, note, tmp_path / "cases", tmp_path / "log.jsonl")
    assert case.decision is None


def test_the_wording_check_catches_banned_words_in_any_case():
    with pytest.raises(GuardrailError):
        assert_clean_wording("A Criminal pattern")
    assert_clean_wording("Refer for review")


def test_the_case_note_lists_the_row_ids_behind_each_finding(tmp_path):
    _, case = run(tmp_path)
    assert "just under the reporting limit within 2 days, totalling ₹91,000. (rows: A-1-T001, A-1-T002)" in case.note
    money_line = next(line for line in case.note.splitlines() if line.startswith("- Money in was"))
    assert "(rows" not in money_line


def test_step_three_summary_and_details_do_not_change_when_rows_are_added(tmp_path):
    _, case = run(tmp_path)
    assert case.steps[2].summary == "Computed 2 findings."
    assert case.steps[2].details == build_findings(make_alert().features)
