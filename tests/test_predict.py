"""Checks answer validation and the retry logic, using fake models (no GPU)."""
import json

from src.predict import classify_with_retry
from src.schema import check_answer

SUMMARY = "7 cash deposits of ₹45,000 each were made within 5 days."
GOOD = json.dumps({"typology": "STRUCTURING", "risk": "high",
                   "reason": "7 deposits of ₹45,000 within 5 days."})


def test_a_good_answer_passes_every_check():
    assert check_answer(GOOD, SUMMARY).valid


def test_text_that_is_not_json_fails():
    assert not check_answer("I think it is structuring", SUMMARY).valid


def test_json_inside_extra_text_is_still_read():
    assert check_answer("Answer: " + GOOD + " Thanks", SUMMARY).valid


def test_an_unknown_label_fails():
    bad = json.dumps({"typology": "SMURFING", "risk": "high", "reason": "ok"})
    assert "typology is not one of the allowed labels" in check_answer(bad, SUMMARY).problems


def test_a_number_missing_from_the_summary_fails():
    bad = json.dumps({"typology": "STRUCTURING", "risk": "high", "reason": "12 deposits."})
    assert "reason uses a number that is not in the summary" in check_answer(bad, SUMMARY).problems


def test_accusing_words_fail():
    bad = json.dumps({"typology": "STRUCTURING", "risk": "high", "reason": "This is fraud."})
    assert "reason uses accusing wording" in check_answer(bad, SUMMARY).problems


def test_a_long_reason_fails():
    bad = json.dumps({"typology": "STRUCTURING", "risk": "high", "reason": "x" * 201})
    assert "reason is too long" in check_answer(bad, SUMMARY).problems


def test_an_invalid_first_answer_is_retried_once():
    answers = iter(["nonsense", GOOD])
    assert classify_with_retry(SUMMARY, lambda summary: next(answers)).valid


def test_two_invalid_answers_give_up_after_one_retry():
    calls = []
    def always_bad(summary):
        calls.append(1)
        return "nonsense"
    assert not classify_with_retry(SUMMARY, always_bad).valid
    assert len(calls) == 2


def test_batched_prediction_answers_every_alert_and_retries_only_bad_ones():
    from src.predict import predict_split
    examples = [{"alert_id": f"A{i}", "summary": SUMMARY} for i in range(5)]
    retries = []

    def batch(summaries):
        # the first answer of every batch is bad, the rest are good
        return ["nonsense"] + [GOOD] * (len(summaries) - 1)

    def single(summary):
        retries.append(1)
        return GOOD

    rows = predict_split(examples, single, batch, batch_size=2)     # batches of 2, 2 and 1
    assert [row["alert_id"] for row in rows] == [f"A{i}" for i in range(5)]
    assert all(row["valid"] for row in rows)
    assert len(retries) == 3                                        # one bad answer per batch
