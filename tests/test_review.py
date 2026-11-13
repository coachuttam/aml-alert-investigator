"""Tests for the data behind the review screen."""
from __future__ import annotations

from src.review import (NEEDS_REVIEW, is_shared_demo, ReviewAlert, agreement_label, build_review_alert,
                        sort_for_queue, words_for)
from src.rules import DEFAULT_CONFIG

STRUCTURING_FEATURES = {
    "just_under_count": 5, "sender_count": 0, "cash_out_percent": 0,
    "return_count": 0, "large_count": 0, "quick_exit_percent": 0,
}


def make_example(features=None) -> dict:
    return {
        "alert_id": "A-test-0001", "customer_id": "C-test-0001", "split": "test",
        "summary": "Five cash deposits just under the limit.",
        "features": features or STRUCTURING_FEATURES,
        "label": {"typology": "STRUCTURING", "risk": "high", "reason": "x"},
        "hard_negative_kind": None, "template_id": "digest",
    }


def make_answer(typology="STRUCTURING", valid=True) -> dict:
    return {"alert_id": "A-test-0001", "valid": valid, "typology": typology,
            "risk": "high", "reason": "5 cash deposits.", "problems": [] if valid else ["bad"]}


def make_alert(**changes) -> ReviewAlert:
    example = make_example()
    alert = build_review_alert(example, DEFAULT_CONFIG, make_answer(), make_answer("NO_CONCERN"), "2026-06-01")
    return ReviewAlert(**{**alert.__dict__, **changes})


def test_alert_joins_the_model_answer_the_rules_and_the_true_label():
    alert = make_alert()
    assert alert.model_typology == "STRUCTURING"
    assert alert.rules_tuned == "STRUCTURING"
    assert alert.base_typology == "NO_CONCERN"
    assert alert.true_typology == "STRUCTURING"


def test_an_invalid_model_answer_is_shown_as_needs_review():
    alert = build_review_alert(make_example(), DEFAULT_CONFIG, make_answer(valid=False), None, "")
    assert alert.model_typology == NEEDS_REVIEW
    assert alert.model_risk == "unknown"
    assert alert.model_reason == ""


def test_a_missing_model_answer_is_shown_as_needs_review():
    alert = build_review_alert(make_example(), DEFAULT_CONFIG, None, None, "")
    assert alert.model_typology == NEEDS_REVIEW


def test_queue_puts_high_risk_first_then_newest_window():
    low = make_alert(alert_id="low", model_risk="low", window_end="2026-09-01")
    old_high = make_alert(alert_id="old_high", model_risk="high", window_end="2026-03-01")
    new_high = make_alert(alert_id="new_high", model_risk="high", window_end="2026-08-01")
    unknown = make_alert(alert_id="unknown", model_risk="unknown", window_end="2026-10-01")
    ordered = [alert.alert_id for alert in sort_for_queue([low, old_high, unknown, new_high])]
    assert ordered == ["new_high", "old_high", "low", "unknown"]


def test_agreement_line_says_when_the_model_and_rules_disagree():
    assert "agree" in agreement_label(make_alert())
    assert "disagree" in agreement_label(make_alert(rules_tuned="NO_CONCERN"))
    assert "could not be validated" in agreement_label(make_alert(model_typology=NEEDS_REVIEW))


def test_typology_names_become_plain_words():
    assert words_for("RAPID_PASS_THROUGH") == "Rapid pass-through"
    assert words_for(NEEDS_REVIEW) == "Needs review"


def test_the_app_text_never_uses_accusing_words():
    from pathlib import Path
    from src.schema import BANNED_WORDS
    root = Path(__file__).resolve().parent.parent
    for name in ("app.py", "evidence_view.py", "upload_view.py"):
        screen_text = (root / name).read_text(encoding="utf-8").lower()
        for word in BANNED_WORDS:
            assert word not in screen_text


def test_a_public_host_is_recognised_so_the_shared_demo_notice_can_show(tmp_path):
    assert is_shared_demo({"SPACE_ID": "someone/space"}, tmp_path / "missing")
    assert is_shared_demo({}, tmp_path)                       # the cloud folder exists
    assert not is_shared_demo({}, tmp_path / "missing")       # a laptop
