"""Checks the numbers, the summaries and the labels built from them."""
import re
from dataclasses import replace
from datetime import date, timedelta

import pytest

from src.dataset import build_dataset, build_example, load_split, save_dataset
from src.features import (HIGH_RATIO, MEDIUM_RATIO, compute_features, risk_level)
from src.generator import generate_all
from src.models import Customer, RawAlert
from src.money import format_rupees, plural
from src.patterns import Ledger
from src.schema import ALL_LABELS, NO_CONCERN, RISK_LEVELS, SPLITS, TYPOLOGIES
from src.templates import HELDOUT_TEMPLATES, SHARED_TEMPLATES, TEMPLATES, section_keys

START = date(2026, 5, 1)
BANNED_WORDS = ("fraud", "criminal", "guilty")


def day(offset):
    """A date `offset` days after the start of the window."""
    return START + timedelta(days=offset)


def make_alert(ledger, typology=NO_CONCERN, income=50_000, age_months=36):
    """Wrap a hand-built ledger into an alert, so features can be tested on tiny cases."""
    customer = Customer("C-test", "freelancer", income, age_months)
    return RawAlert("A-test", "train", "shared", customer, day(0), day(29),
                    ledger.finish(), typology)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def test_money_uses_indian_digit_grouping():
    assert format_rupees(0) == "₹0"
    assert format_rupees(999) == "₹999"
    assert format_rupees(1_000) == "₹1,000"
    assert format_rupees(102_000) == "₹1,02,000"
    assert format_rupees(300_000) == "₹3,00,000"
    assert format_rupees(12_345_678) == "₹1,23,45,678"
    assert format_rupees(-1_500) == "-₹1,500"


def test_plural_agrees_with_the_count():
    assert plural(1, "deposit") == "1 deposit"
    assert plural(4, "deposit") == "4 deposits"
    assert plural(0, "deposit") == "0 deposits"


# --------------------------------------------------------------------------
# The maths, on tiny hand-built alerts
# --------------------------------------------------------------------------

def test_deposits_just_under_the_limit_are_counted_with_their_span():
    ledger = Ledger("A-test")
    for offset in (3, 4, 6):
        ledger.credit(day(offset), 45_000, "cash", "Cash deposit")
    ledger.credit(day(10), 5_000, "cash", "Cash deposit")        # ordinary, far below the limit
    features = compute_features(make_alert(ledger))
    assert features.cash_count == 4
    assert features.just_under_count == 3
    assert features.just_under_total == 135_000
    assert features.just_under_span_days == 4                   # day 3 to day 6, both included


def test_a_large_credit_sent_on_within_two_days_is_a_quick_exit():
    ledger = Ledger("A-test")
    ledger.credit(day(5), 200_000, "neft", "Sharma Traders")
    ledger.debit(day(6), 190_000, "neft", "Patel Stores")
    features = compute_features(make_alert(ledger))
    assert features.large_count == 1
    assert features.quick_exit_amount == 190_000
    assert features.quick_exit_percent == 95


def test_money_sent_after_the_two_day_limit_is_not_a_quick_exit():
    ledger = Ledger("A-test")
    ledger.credit(day(5), 200_000, "neft", "Sharma Traders")
    ledger.debit(day(9), 190_000, "neft", "Patel Stores")
    assert compute_features(make_alert(ledger)).quick_exit_percent == 0


def test_money_moved_to_the_customers_own_deposit_is_not_counted_as_sent_on():
    ledger = Ledger("A-test")
    ledger.credit(day(5), 3_000_000, "rtgs", "Ravi Kumar")
    ledger.debit(day(6), 2_000_000, "internal", "Own fixed deposit")
    assert compute_features(make_alert(ledger)).quick_exit_percent == 0


def test_a_burst_of_senders_is_found_even_when_regular_payouts_continue_all_month():
    ledger = Ledger("A-test")
    for week in range(4):                                       # steady weekly income all month
        ledger.credit(day(week * 7 + 1), 5_000, "upi", "Delivery platform payout")
    for number in range(10):                                    # ten different senders in three days
        ledger.credit(day(10 + number % 3), 10_000, "upi", f"Sender {number}")
    ledger.debit(day(13), 90_000, "neft", "Ravi Kumar")         # cashed out the day after the burst
    features = compute_features(make_alert(ledger))
    assert features.sender_count >= 10
    assert features.cash_out_percent >= 85


def test_a_transfer_that_comes_back_nearly_whole_is_a_returned_payment():
    ledger = Ledger("A-test")
    ledger.debit(day(2), 100_000, "neft", "Sharma Traders")
    ledger.credit(day(7), 97_000, "neft", "Patel Stores")
    features = compute_features(make_alert(ledger))
    assert features.return_count == 1
    assert features.return_total == 97_000
    assert features.return_same_party == 0


def test_a_return_from_the_same_person_is_counted_as_the_same_party():
    ledger = Ledger("A-test")
    ledger.debit(day(2), 20_000, "upi", "Ravi Kumar")
    ledger.credit(day(9), 20_000, "upi", "Ravi Kumar")
    assert compute_features(make_alert(ledger)).return_same_party == 1


def test_cash_withdrawals_and_deposits_are_never_treated_as_returned_payments():
    ledger = Ledger("A-test")
    ledger.debit(day(2), 10_000, "atm", "ATM withdrawal")
    ledger.credit(day(5), 9_500, "upi", "Customer receipts")
    ledger.debit(day(8), 30_000, "neft", "Sharma Traders")
    ledger.credit(day(11), 30_000, "cash", "Cash deposit")
    assert compute_features(make_alert(ledger)).return_count == 0


def test_money_that_comes_back_after_fifteen_days_is_not_a_return():
    ledger = Ledger("A-test")
    ledger.debit(day(1), 50_000, "neft", "Sharma Traders")
    ledger.credit(day(20), 50_000, "neft", "Patel Stores")
    assert compute_features(make_alert(ledger)).return_count == 0


# --------------------------------------------------------------------------
# Risk level
# --------------------------------------------------------------------------

def structuring_features_with_ratio(ratio):
    features = compute_features(make_alert(Ledger("A-test")))
    return replace(features, just_under_ratio=ratio)


def test_risk_depends_only_on_the_amount_compared_with_monthly_income():
    just_below_medium = structuring_features_with_ratio(MEDIUM_RATIO - 0.1)
    at_medium = structuring_features_with_ratio(MEDIUM_RATIO)
    just_below_high = structuring_features_with_ratio(HIGH_RATIO - 0.1)
    at_high = structuring_features_with_ratio(HIGH_RATIO)
    assert risk_level("STRUCTURING", just_below_medium) == "low"
    assert risk_level("STRUCTURING", at_medium) == "medium"
    assert risk_level("STRUCTURING", just_below_high) == "medium"
    assert risk_level("STRUCTURING", at_high) == "high"


def test_normal_activity_is_always_low_risk():
    features = structuring_features_with_ratio(99.0)
    assert risk_level(NO_CONCERN, features) == "low"


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------

def test_every_template_defines_every_piece_of_wording():
    expected_keys = set(TEMPLATES["digest"])
    assert set(TEMPLATES) == set(SHARED_TEMPLATES) | set(HELDOUT_TEMPLATES)
    for name, template in TEMPLATES.items():
        assert set(template) == expected_keys, f"{name} is missing or has extra wording"


def test_the_section_list_always_covers_profile_to_totals():
    features = compute_features(make_alert(Ledger("A-test")))
    keys = section_keys(features)
    assert keys[0] == "profile" and keys[-1] == "totals" and len(keys) == 7


# --------------------------------------------------------------------------
# The whole dataset
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_alerts():
    return generate_all()


@pytest.fixture(scope="module")
def dataset(raw_alerts):
    return build_dataset(raw_alerts)


def all_examples(dataset):
    return [example for examples in dataset.values() for example in examples]


def numbers_in(text):
    """Every number in the text, with thousands separators removed."""
    return {match.replace(",", "") for match in re.findall(r"\d[\d,]*", text)}


def test_every_reason_uses_only_numbers_found_in_its_summary(dataset):
    for example in all_examples(dataset):
        missing = numbers_in(example["label"]["reason"]) - numbers_in(example["summary"])
        assert not missing, f"{example['alert_id']}: reason uses {missing} which the summary does not show"


def test_no_summary_or_reason_uses_accusing_words(dataset):
    for example in all_examples(dataset):
        text = (example["summary"] + " " + example["label"]["reason"]).lower()
        for word in BANNED_WORDS:
            assert word not in text


def test_summaries_never_give_the_number_of_transactions(dataset):
    for example in all_examples(dataset):
        assert "transaction" not in example["summary"].lower()


def test_heldout_uses_only_reserved_templates_and_no_other_split_does(dataset):
    for split in SPLITS:
        templates = {example["template_id"] for example in dataset[split]}
        if split == "heldout":
            assert templates == set(HELDOUT_TEMPLATES)
        else:
            assert templates == set(SHARED_TEMPLATES)


def test_every_label_is_valid_and_every_reason_is_short(dataset):
    for example in all_examples(dataset):
        label = example["label"]
        assert label["typology"] in ALL_LABELS
        assert label["risk"] in RISK_LEVELS
        assert 0 < len(label["reason"]) <= 200


def test_the_training_set_has_every_risk_level_for_every_pattern(dataset):
    for typology in TYPOLOGIES:
        levels = {e["label"]["risk"] for e in dataset["train"] if e["label"]["typology"] == typology}
        assert levels == set(RISK_LEVELS), f"{typology} has only {levels}"


def test_each_pattern_is_visible_in_the_numbers(dataset):
    """If the numbers do not show the pattern, no model could be expected to find it."""
    for example in all_examples(dataset):
        f, typology = example["features"], example["label"]["typology"]
        if typology == "STRUCTURING":
            assert f["just_under_count"] >= 6
        elif typology == "RAPID_PASS_THROUGH":
            assert f["large_count"] >= 2 and f["quick_exit_percent"] >= 80
        elif typology == "MULE_ACCOUNT":
            assert f["sender_count"] >= 10
        elif typology == "ROUND_TRIPPING":
            assert f["return_count"] >= 2


def test_most_mule_alerts_show_a_quick_cash_out(dataset):
    mules = [e["features"] for e in all_examples(dataset) if e["label"]["typology"] == "MULE_ACCOUNT"]
    quick = [f for f in mules if f["cash_out_percent"] >= 50]
    assert len(quick) >= 0.95 * len(mules)


def test_building_the_dataset_twice_gives_identical_examples(raw_alerts, dataset):
    assert build_dataset(raw_alerts) == dataset


def test_one_example_is_built_from_one_alert(raw_alerts):
    raw = raw_alerts["train"][0]
    example = build_example(raw)
    assert example["alert_id"] == raw.alert_id
    assert example["label"]["typology"] == raw.typology


def test_a_saved_dataset_loads_back_unchanged(dataset, tmp_path):
    save_dataset(dataset, tmp_path)
    for split in SPLITS:
        assert load_split(tmp_path / f"{split}.jsonl") == dataset[split]
