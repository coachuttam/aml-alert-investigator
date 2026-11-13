"""Checks that each finding points at the right transaction rows."""
from dataclasses import asdict
from datetime import date, timedelta
from functools import lru_cache

from src.case_note import build_findings
from src.evidence import (EVIDENCE_KINDS, build_findings_with_rows, cited_ids, findings_for_rows,
                          rows_for)
from src.review import read_transactions
from src.features import compute_features
from src.generator import generate_all
from src.models import Customer, RawAlert, Transaction

TYPOLOGY_FOR_KIND = {"just_under": "STRUCTURING", "large_credits": "RAPID_PASS_THROUGH",
                     "senders": "MULE_ACCOUNT", "returns": "ROUND_TRIPPING"}


@lru_cache(maxsize=1)
def training_alerts() -> list:
    """Training-split alerts only; the held-out set is never touched here."""
    return generate_all()["train"]


def alerts_with(typology: str) -> list:
    """A few training alerts that carry the given planted pattern."""
    return [a for a in training_alerts() if a.typology == typology][:15]


def features_of(alert: RawAlert) -> dict:
    return asdict(compute_features(alert))


def test_just_under_rows_match_the_just_under_count():
    for alert in alerts_with("STRUCTURING"):
        features = features_of(alert)
        assert len(rows_for("just_under", alert.transactions, features)) == features["just_under_count"]


def test_large_credit_rows_start_with_one_row_per_large_credit():
    for alert in alerts_with("RAPID_PASS_THROUGH"):
        features = features_of(alert)
        rows = rows_for("large_credits", alert.transactions, features)
        by_id = {t.txn_id: t for t in alert.transactions}
        credit_rows = [r for r in rows if by_id[r].direction == "credit"]
        assert len(credit_rows) == features["large_count"]


def test_sender_rows_cover_every_sender_counted():
    for alert in alerts_with("MULE_ACCOUNT"):
        features = features_of(alert)
        rows = rows_for("senders", alert.transactions, features)
        by_id = {t.txn_id: t for t in alert.transactions}
        assert len({by_id[r].counterparty for r in rows}) == features["sender_count"]


def test_return_rows_match_the_return_count():
    for alert in alerts_with("ROUND_TRIPPING"):
        features = features_of(alert)
        assert len(rows_for("returns", alert.transactions, features)) == features["return_count"]


def test_every_returned_id_exists_in_the_alert():
    for typology, kind in ((v, k) for k, v in TYPOLOGY_FOR_KIND.items()):
        for alert in alerts_with(typology):
            ids = {t.txn_id for t in alert.transactions}
            assert set(rows_for(kind, alert.transactions, features_of(alert))) <= ids


def test_an_alert_with_no_planted_pattern_returns_no_rows():
    start = date(2026, 5, 1)
    transactions = [
        Transaction("T1", start, 30_000, "credit", "salary", "Employer"),
        Transaction("T2", start + timedelta(days=3), 900, "debit", "upi", "Grocery store"),
        Transaction("T3", start + timedelta(days=9), 12_000, "debit", "neft", "Landlord"),
    ]
    customer = Customer("C-test", "salaried", 30_000, 36)
    alert = RawAlert("A-test", "train", "shared", customer, start, start + timedelta(days=29),
                     transactions, "NO_CONCERN")
    features = features_of(alert)
    for kind in EVIDENCE_KINDS:
        assert rows_for(kind, alert.transactions, features) == []


def test_no_rows_are_returned_when_the_matching_feature_count_is_zero():
    count_for_kind = {"just_under": "just_under_count", "large_credits": "large_count",
                      "returns": "return_count"}
    for alert in training_alerts()[:300]:
        features = features_of(alert)
        for kind, count_name in count_for_kind.items():
            if features[count_name] == 0:
                assert rows_for(kind, alert.transactions, features) == []


def test_findings_with_rows_keep_the_wording_and_order_of_build_findings():
    for alert in alerts_with("STRUCTURING") + alerts_with("MULE_ACCOUNT"):
        features = features_of(alert)
        texts = [text for text, _ in build_findings_with_rows(features, alert.transactions)]
        assert texts == build_findings(features)


def test_findings_from_screen_rows_match_findings_from_transactions():
    alert = alerts_with("MULE_ACCOUNT")[0]
    rows = [{"txn_id": t.txn_id, "day": t.day.isoformat(), "amount": t.amount, "direction": t.direction,
             "channel": t.channel, "counterparty": t.counterparty} for t in alert.transactions]
    features = features_of(alert)
    assert findings_for_rows(features, rows) == build_findings_with_rows(features, alert.transactions)


def test_mule_and_round_trip_rows_are_cited_so_the_table_marks_them():
    for typology in ("MULE_ACCOUNT", "ROUND_TRIPPING"):
        alert = alerts_with(typology)[0]
        findings = build_findings_with_rows(features_of(alert), alert.transactions)
        assert cited_ids(findings)
