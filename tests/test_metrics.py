"""Checks the scoring maths on tiny hand-made cases."""
import pytest

from src.metrics import false_positive_rates, macro_f1, safe_divide, score_label

STRUCT, MULE, NONE = "STRUCTURING", "MULE_ACCOUNT", "NO_CONCERN"


def test_dividing_by_zero_gives_zero():
    assert safe_divide(3, 0) == 0.0


def test_precision_recall_and_f1_for_one_label():
    true = [STRUCT, STRUCT, NONE, NONE]
    guess = [STRUCT, NONE, STRUCT, NONE]
    score = score_label(true, guess, STRUCT)
    assert score.precision == 0.5 and score.recall == 0.5 and score.f1 == 0.5


def test_perfect_predictions_give_macro_f1_of_one_over_labels_present():
    true = [STRUCT, MULE, NONE]
    # labels never seen score 0, so a perfect run on 3 of 5 labels is 3/5
    assert macro_f1(true, true) == pytest.approx(3 / 5)


def test_false_positive_rate_is_split_by_kind_and_ignores_real_patterns():
    true = [NONE, NONE, NONE, STRUCT]
    guess = [STRUCT, NONE, STRUCT, STRUCT]
    kinds = [None, None, "house_sale", None]
    rates = false_positive_rates(true, guess, kinds)
    assert rates["ordinary"] == 0.5
    assert rates["house_sale"] == 1.0
    assert rates["all_no_concern"] == pytest.approx(2 / 3)
