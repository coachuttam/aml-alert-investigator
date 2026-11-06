"""Plain data containers shared by every module."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional


@dataclass(frozen=True)
class Transaction:
    """One money movement on the customer's account."""

    txn_id: str
    day: date
    amount: int         # whole rupees, always positive
    direction: str      # "credit" (money in) or "debit" (money out)
    channel: str        # cash, upi, imps, neft, rtgs, atm, salary or internal
    counterparty: str   # who sent or received the money


@dataclass(frozen=True)
class Customer:
    """The profile an analyst would see next to the alert."""

    customer_id: str
    occupation: str
    declared_monthly_income: int   # whole rupees
    account_age_months: int


@dataclass
class RawAlert:
    """One customer's 30-day window with the true answer attached.

    This is the generator's output. Features and the alert summary are added
    in a later step, which turns a RawAlert into a training example.
    """

    alert_id: str
    split: str                      # one of schema.SPLITS
    template_group: str             # "shared" or "heldout" (which wording templates may be used)
    customer: Customer
    window_start: date
    window_end: date
    transactions: List[Transaction]
    typology: str                   # one of schema.ALL_LABELS
    hard_negative_kind: Optional[str] = None   # set for legitimate look-alikes
