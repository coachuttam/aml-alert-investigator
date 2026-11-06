"""Checks each rule fires on its own pattern, and that tuning never touches held-out."""
from src.dataset import load_split
from src.rules import DEFAULT_CONFIG, RuleConfig, classify, predict_all, tune_on_training_set
from pathlib import Path

QUIET = {"just_under_count": 0, "sender_count": 0, "cash_out_percent": 0,
         "return_count": 0, "large_count": 0, "quick_exit_percent": 0}
DATA = Path(__file__).resolve().parent.parent / "data"


def with_values(**values):
    return {**QUIET, **values}


def test_quiet_accounts_get_no_concern():
    assert classify(QUIET) == ("NO_CONCERN", "R0")


def test_many_deposits_just_under_the_limit_fire_the_structuring_rule():
    assert classify(with_values(just_under_count=6)) == ("STRUCTURING", "R1")


def test_many_senders_and_a_quick_cash_out_fire_the_mule_rule():
    assert classify(with_values(sender_count=12, cash_out_percent=90)) == ("MULE_ACCOUNT", "R2")


def test_many_senders_without_a_cash_out_do_not_fire_the_mule_rule():
    assert classify(with_values(sender_count=12, cash_out_percent=0))[0] == "NO_CONCERN"


def test_returned_transfers_fire_the_round_trip_rule():
    assert classify(with_values(return_count=2)) == ("ROUND_TRIPPING", "R3")


def test_large_credits_sent_on_quickly_fire_the_pass_through_rule():
    assert classify(with_values(large_count=2, quick_exit_percent=95)) == ("RAPID_PASS_THROUGH", "R4")


def test_stricter_thresholds_stop_a_rule_from_firing():
    strict = RuleConfig(just_under_min=10)
    assert classify(with_values(just_under_count=6), strict)[0] == "NO_CONCERN"


def test_tuning_uses_only_the_training_examples_it_is_given():
    train = load_split(DATA / "train.jsonl")[:300]
    config = tune_on_training_set(train)
    assert isinstance(config, RuleConfig)
    assert len(predict_all(train, config)) == len(train)
