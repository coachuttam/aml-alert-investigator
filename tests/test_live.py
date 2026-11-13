"""Tests for the live run: reading an upload, asking the model and failing safely."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.live import MODEL_NOT_CONNECTED, build_live_alert, new_upload_id
from src.models import Customer
from src.remote_model import ModelUnavailable, live_generate
from src.review import NEEDS_REVIEW
from src.rules import DEFAULT_CONFIG
from src.upload import MAX_ROWS, NOT_KNOWN, UploadError, read_upload, transaction_rows

SAMPLE = (Path(__file__).resolve().parent.parent / "data" / "sample_upload.csv").read_text(encoding="utf-8")
CUSTOMER = Customer("C-upload", "delivery driver", 31000, 34)
HEADER = "day,amount,direction,channel,counterparty\n"
GOOD_ANSWER = '{"typology": "STRUCTURING", "risk": "high", "reason": "Cash deposits just under the limit."}'


def read_sample():
    return read_upload(SAMPLE, "U-1", CUSTOMER)


def test_sample_file_reads_into_an_alert_with_no_true_label():
    raw = read_sample()
    assert len(raw.transactions) == 26 and raw.typology == NOT_KNOWN
    assert raw.window_start <= raw.window_end
    assert transaction_rows(raw)[0]["day"] == raw.window_start.isoformat()


def test_amounts_with_commas_and_mixed_case_columns_are_accepted():
    raw = read_upload("Day,Amount,Direction,Channel,Counterparty\n2026-09-01,\"48,000\",Credit,CASH,Cash deposit\n",
                      "U-1", CUSTOMER)
    assert raw.transactions[0].amount == 48000 and raw.transactions[0].channel == "cash"


@pytest.mark.parametrize("text, words", [
    ("day,amount\n2026-09-01,5\n", "missing these columns"),
    (HEADER, "no transactions"),
    (HEADER + "01/09/2026,5,credit,cash,X\n", "Row 2: day"),
    (HEADER + "2026-09-01,-5,credit,cash,X\n", "Row 2: amount"),
    (HEADER + "2026-09-01,5,in,cash,X\n", "Row 2: direction"),
    (HEADER + "2026-09-01,5,credit,cheque,X\n", "Row 2: channel"),
    (HEADER + "2026-09-01,5,credit,cash,\n", "Row 2: counterparty"),
    (HEADER + "2026-09-01,5,credit,cash,X\n" * (MAX_ROWS + 1), "the limit is"),
])
def test_a_bad_file_is_refused_with_a_plain_message(text, words):
    with pytest.raises(UploadError, match=words):
        read_upload(text, "U-1", CUSTOMER)


def test_zero_income_is_refused():
    with pytest.raises(UploadError, match="income"):
        read_upload(SAMPLE, "U-1", Customer("C-upload", "driver", 0, 5))


def test_a_valid_live_answer_is_shown_like_a_saved_one():
    alert = build_live_alert(read_sample(), lambda summary: GOOD_ANSWER, DEFAULT_CONFIG)
    assert (alert.model_typology, alert.model_risk) == ("STRUCTURING", "high")
    assert alert.rules_default == "STRUCTURING" and alert.true_typology == NOT_KNOWN


def test_an_invented_number_is_retried_once_then_sent_to_review():
    calls = []

    def invents_a_number(summary: str) -> str:
        calls.append(summary)
        return '{"typology": "STRUCTURING", "risk": "high", "reason": "987654321 rupees moved."}'

    alert = build_live_alert(read_sample(), invents_a_number, DEFAULT_CONFIG)
    assert len(calls) == 2 and alert.model_typology == NEEDS_REVIEW
    assert "reason uses a number that is not in the summary" in alert.model_problems


def test_no_model_or_an_unreachable_model_goes_to_review_and_rules_still_run():
    def unreachable(summary: str) -> str:
        raise ModelUnavailable("the live model did not answer (timed out)")

    not_connected = build_live_alert(read_sample(), None, DEFAULT_CONFIG)
    timed_out = build_live_alert(read_sample(), unreachable, DEFAULT_CONFIG)
    assert not_connected.model_problems == [MODEL_NOT_CONNECTED]
    assert timed_out.model_typology == NEEDS_REVIEW and "did not answer" in timed_out.model_problems[0]
    assert timed_out.rules_default == "STRUCTURING"


def test_live_model_is_off_unless_a_secure_address_is_set():
    assert live_generate({}) is None
    assert live_generate({"AML_MODEL_URL": "http://not-secure.example"}) is None
    assert live_generate({"AML_MODEL_URL": "https://model.example/classify"}) is not None


def test_upload_ids_are_readable():
    assert new_upload_id().startswith("U-20")
