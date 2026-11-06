"""Checks that the generator plants exactly the patterns it claims to, with no leaks."""
from collections import Counter
from datetime import timedelta

import pytest

from src.generator import SPLIT_PLANS, generate_all, load_raw_alerts, save_raw_alerts
from src.names import HELDOUT_NAMES, STANDARD_NAMES
from src.patterns import CASH_REPORTING_LIMIT, JUST_UNDER_HIGH, JUST_UNDER_LOW
from src.schema import NO_CONCERN, SPLITS, TYPOLOGIES


@pytest.fixture(scope="module")
def alerts_by_split():
    """Generate every split once and share it across the tests."""
    return generate_all()


def all_alerts(alerts_by_split):
    return [alert for alerts in alerts_by_split.values() for alert in alerts]


def alerts_with_typology(alerts_by_split, typology):
    return [alert for alert in all_alerts(alerts_by_split) if alert.typology == typology]


def just_under_limit_deposits(alert):
    """Cash deposits between 80% and 99% of the reporting limit."""
    low = JUST_UNDER_LOW * CASH_REPORTING_LIMIT
    high = JUST_UNDER_HIGH * CASH_REPORTING_LIMIT
    return [t for t in alert.transactions
            if t.direction == "credit" and t.channel == "cash" and low <= t.amount <= high]


def test_same_seed_gives_identical_alerts(alerts_by_split):
    again = generate_all()
    assert again == alerts_by_split


def test_split_sizes_match_the_plan(alerts_by_split):
    sizes = {split: len(alerts) for split, alerts in alerts_by_split.items()}
    assert sizes == {"train": 2400, "validation": 300, "test": 400, "heldout": 300}


def test_label_counts_match_the_plan_in_every_split(alerts_by_split):
    for split in SPLITS:
        plan = SPLIT_PLANS[split]
        counts = Counter(alert.typology for alert in alerts_by_split[split])
        for typology in TYPOLOGIES:
            assert counts[typology] == plan.per_typology
        assert counts[NO_CONCERN] == plan.benign + plan.hard_negatives


def test_hard_negatives_are_shared_equally_between_the_four_kinds(alerts_by_split):
    for split in SPLITS:
        kinds = Counter(a.hard_negative_kind for a in alerts_by_split[split] if a.hard_negative_kind)
        assert len(kinds) == 4
        assert set(kinds.values()) == {SPLIT_PLANS[split].hard_negatives // 4}


def test_every_customer_belongs_to_exactly_one_split(alerts_by_split):
    seen = {}
    for alert in all_alerts(alerts_by_split):
        customer_id = alert.customer.customer_id
        assert customer_id not in seen, f"{customer_id} appears twice"
        seen[customer_id] = alert.split
    assert len(seen) == 3400


def test_all_amounts_are_positive_whole_rupees_inside_the_window(alerts_by_split):
    for alert in all_alerts(alerts_by_split):
        assert alert.window_end - alert.window_start == timedelta(days=29)
        for txn in alert.transactions:
            assert isinstance(txn.amount, int) and txn.amount > 0
            assert alert.window_start <= txn.day <= alert.window_end


def test_transactions_are_in_date_order_with_unique_ids(alerts_by_split):
    for alert in all_alerts(alerts_by_split):
        days = [t.day for t in alert.transactions]
        assert days == sorted(days)
        ids = [t.txn_id for t in alert.transactions]
        assert len(set(ids)) == len(ids)


def test_structuring_alerts_have_several_deposits_just_under_the_limit(alerts_by_split):
    for alert in alerts_with_typology(alerts_by_split, "STRUCTURING"):
        assert len(just_under_limit_deposits(alert)) >= 6


def test_no_concern_alerts_never_have_deposits_just_under_the_limit(alerts_by_split):
    for alert in alerts_with_typology(alerts_by_split, NO_CONCERN):
        assert just_under_limit_deposits(alert) == []


def test_pass_through_alerts_send_most_of_each_large_inflow_out_within_two_days(alerts_by_split):
    for alert in alerts_with_typology(alerts_by_split, "RAPID_PASS_THROUGH"):
        quick_exits = 0
        for inflow in [t for t in alert.transactions if t.direction == "credit" and t.amount >= 150_000]:
            for outflow in [t for t in alert.transactions if t.direction == "debit"]:
                within_two_days = timedelta(0) <= outflow.day - inflow.day <= timedelta(days=2)
                if within_two_days and outflow.amount >= 0.89 * inflow.amount:
                    quick_exits += 1
        assert quick_exits >= 2


def test_mule_alerts_have_many_different_senders_and_a_new_account(alerts_by_split):
    for alert in alerts_with_typology(alerts_by_split, "MULE_ACCOUNT"):
        senders = {t.counterparty for t in alert.transactions
                   if t.direction == "credit" and t.channel in ("upi", "imps")}
        assert len(senders) >= 10
        assert alert.customer.account_age_months <= 6


def test_round_tripping_alerts_get_the_money_back_within_fifteen_days(alerts_by_split):
    for alert in alerts_with_typology(alerts_by_split, "ROUND_TRIPPING"):
        returned = 0
        for outflow in [t for t in alert.transactions if t.direction == "debit" and t.amount >= 100_000]:
            for inflow in [t for t in alert.transactions if t.direction == "credit"]:
                within_fifteen_days = timedelta(0) < inflow.day - outflow.day <= timedelta(days=15)
                if within_fifteen_days and inflow.amount >= 0.94 * outflow.amount:
                    returned += 1
        assert returned >= 2


def test_the_repay_friend_look_alike_returns_the_money_to_the_same_person(alerts_by_split):
    look_alikes = [a for a in all_alerts(alerts_by_split) if a.hard_negative_kind == "repay_friend"]
    assert look_alikes
    for alert in look_alikes:
        credits_from = {(t.counterparty, t.amount) for t in alert.transactions
                        if t.direction == "credit" and t.channel == "upi"}
        debits_to = {(t.counterparty, t.amount) for t in alert.transactions
                     if t.direction == "debit" and t.channel == "upi"}
        assert credits_from & debits_to


def test_the_house_sale_look_alike_moves_money_to_the_customers_own_fixed_deposit(alerts_by_split):
    look_alikes = [a for a in all_alerts(alerts_by_split) if a.hard_negative_kind == "house_sale"]
    assert look_alikes
    for alert in look_alikes:
        assert any(t.counterparty == "Own fixed deposit" for t in alert.transactions)


def test_the_name_pools_share_no_names():
    assert not set(STANDARD_NAMES.first_names) & set(HELDOUT_NAMES.first_names)
    assert not set(STANDARD_NAMES.last_names) & set(HELDOUT_NAMES.last_names)
    assert not set(STANDARD_NAMES.business_endings) & set(HELDOUT_NAMES.business_endings)


def test_heldout_alerts_never_use_a_name_from_the_training_pool(alerts_by_split):
    training_people = set(STANDARD_NAMES.all_people())
    for alert in alerts_by_split["heldout"]:
        counterparties = {t.counterparty for t in alert.transactions}
        assert not counterparties & training_people


def test_heldout_mule_senders_are_all_people_from_the_heldout_name_pool(alerts_by_split):
    heldout_people = set(HELDOUT_NAMES.all_people())
    for alert in alerts_by_split["heldout"]:
        if alert.typology != "MULE_ACCOUNT":
            continue
        senders = {t.counterparty for t in alert.transactions
                   if t.direction == "credit" and t.counterparty in heldout_people}
        assert len(senders) >= 10


def test_heldout_amounts_are_larger_than_training_amounts(alerts_by_split):
    def median_large_inflow(alerts):
        amounts = sorted(t.amount for a in alerts if a.typology == "RAPID_PASS_THROUGH"
                         for t in a.transactions if t.direction == "credit" and t.amount >= 150_000)
        return amounts[len(amounts) // 2]
    assert median_large_inflow(alerts_by_split["heldout"]) > 1.3 * median_large_inflow(alerts_by_split["train"])


def test_heldout_alerts_are_tagged_with_the_reserved_template_group(alerts_by_split):
    assert {a.template_group for a in alerts_by_split["heldout"]} == {"heldout"}
    for split in ("train", "validation", "test"):
        assert {a.template_group for a in alerts_by_split[split]} == {"shared"}


def test_saved_alerts_load_back_unchanged(alerts_by_split, tmp_path):
    save_raw_alerts(alerts_by_split, tmp_path)
    for split in SPLITS:
        assert load_raw_alerts(tmp_path / f"{split}.jsonl") == alerts_by_split[split]
