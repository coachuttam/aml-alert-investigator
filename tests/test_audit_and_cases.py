"""Tests for the audit log and the case files."""
from __future__ import annotations

from src.audit import append_event, read_events
from src.case_store import AWAITING_ANALYST, Case, StepRecord, count_cases, load_case, save_case


def test_events_are_appended_in_order_and_never_rewritten(tmp_path):
    log_path = tmp_path / "log" / "audit_log.jsonl"
    append_event("A-1", "step_1_load_alert", after="loaded", log_path=log_path)
    append_event("A-1", "decision", before="STRUCTURING / high", after="NO_CONCERN / low",
                 note="  salary pattern  ", log_path=log_path)
    events = read_events(log_path)
    assert [event.action for event in events] == ["step_1_load_alert", "decision"]
    assert events[1].before == "STRUCTURING / high"
    assert events[1].note == "salary pattern"


def test_reading_an_audit_log_that_does_not_exist_gives_an_empty_list(tmp_path):
    assert read_events(tmp_path / "nothing.jsonl") == []


def test_a_saved_case_comes_back_with_its_steps_and_decision(tmp_path):
    steps = [StepRecord(1, "Load alert", "done", "Loaded A-1.", ["line one"])]
    case = Case("A-1", AWAITING_ANALYST, steps, "Draft note.", {"action": "approve"})
    save_case(case, tmp_path)
    loaded = load_case("A-1", tmp_path)
    assert loaded == case
    assert loaded.steps[0].details == ["line one"]


def test_loading_a_case_that_was_never_saved_gives_none(tmp_path):
    assert load_case("A-404", tmp_path) is None


def test_saving_a_case_again_replaces_the_old_file_and_counts_stay_correct(tmp_path):
    save_case(Case("A-1", AWAITING_ANALYST, [], "first"), tmp_path)
    save_case(Case("A-1", AWAITING_ANALYST, [], "second"), tmp_path)
    save_case(Case("A-2", AWAITING_ANALYST, [], "other"), tmp_path)
    assert load_case("A-1", tmp_path).note == "second"
    assert count_cases(tmp_path) == 2
