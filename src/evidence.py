"""Maps each computed finding to the transaction rows behind it.

An analyst who reads "3 cash deposits just under the limit" can then see exactly
which three transactions that sentence is about. Nothing is recomputed here with
new thresholds: the constants and helpers come from features.py and patterns.py.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from src import case_note
from src.features import (LARGE_AMOUNT, MIN_SENDERS, QUICK_EXIT_DAYS, busiest_week,
                          find_returned_pairs)
from src.models import Transaction
from src.patterns import CASH_REPORTING_LIMIT, JUST_UNDER_HIGH, JUST_UNDER_LOW

EVIDENCE_KINDS = ("just_under", "large_credits", "senders", "returns")


def split_by_direction(transactions: List[Transaction]) -> Tuple[List[Transaction], List[Transaction]]:
    """Credits, and debits to other people (money moved to the customer's own deposit is left out)."""
    credits = [t for t in transactions if t.direction == "credit"]
    external_debits = [t for t in transactions if t.direction == "debit" and t.channel != "internal"]
    return credits, external_debits


def just_under_rows(credits: List[Transaction]) -> List[str]:
    """Cash credits close to, but under, the reporting limit."""
    low = JUST_UNDER_LOW * CASH_REPORTING_LIMIT
    high = JUST_UNDER_HIGH * CASH_REPORTING_LIMIT
    return [c.txn_id for c in credits if c.channel == "cash" and low <= c.amount <= high]


def large_credit_rows(credits: List[Transaction], external_debits: List[Transaction]) -> List[str]:
    """Large credits, then the debits to others that followed one within QUICK_EXIT_DAYS."""
    large = [c for c in credits if c.amount >= LARGE_AMOUNT]
    quick_exits = []
    for debit in external_debits:
        if any(0 <= (debit.day - credit.day).days <= QUICK_EXIT_DAYS for credit in large):
            quick_exits.append(debit)
    return [t.txn_id for t in large + quick_exits]


def sender_rows(credits: List[Transaction]) -> List[str]:
    """The UPI or IMPS payments in the busiest 7-day window."""
    paid_in = [c for c in credits if c.channel in ("upi", "imps")]
    return [c.txn_id for c in busiest_week(paid_in)]


def return_rows(credits: List[Transaction], external_debits: List[Transaction]) -> List[str]:
    """The credits that brought back money sent out earlier."""
    return [credit.txn_id for _, credit in find_returned_pairs(credits, external_debits)]


def rows_for(kind: str, transactions: List[Transaction], features: Dict) -> List[str]:
    """The txn_ids behind one kind of finding, in the order the finding counts them.

    Returns an empty list when the finding would not be shown (for example, fewer
    than MIN_SENDERS different senders).
    """
    credits, external_debits = split_by_direction(transactions)
    if kind == "just_under":
        return just_under_rows(credits)
    if kind == "large_credits":
        return large_credit_rows(credits, external_debits)
    if kind == "senders":
        return sender_rows(credits) if features["sender_count"] >= MIN_SENDERS else []
    if kind == "returns":
        return return_rows(credits, external_debits)
    raise ValueError(f"Unknown evidence kind: {kind}")


def build_findings_with_rows(features: Dict, transactions: List[Transaction]) -> List[Tuple[str, List[str]]]:
    """Each finding from case_note.build_findings, paired with its txn_ids.

    The wording and order come straight from case_note, so the two never drift.
    Findings that are not about specific rows (money in and out, "no pattern") get [].
    """
    kind_by_text: Dict[str, str] = {}
    for kind, make_lines in (("just_under", case_note.cash_line), ("large_credits", case_note.pass_through_line),
                             ("senders", case_note.sender_line), ("returns", case_note.return_line)):
        for line in make_lines(features):
            kind_by_text[line] = kind
    pairs = []
    for text in case_note.build_findings(features):
        kind = kind_by_text.get(text)
        pairs.append((text, rows_for(kind, transactions, features) if kind else []))
    return pairs
