"""Computes every number about an alert, plus its risk level and reason.

This is where "Python counts" happens. The model later reads a summary built
from these numbers and never produces a number of its own.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.models import RawAlert, Transaction
from src.money import format_rupees, plural
from src.patterns import CASH_REPORTING_LIMIT, JUST_UNDER_HIGH, JUST_UNDER_LOW
from src.schema import NO_CONCERN

# A single credit this large is worth following to see where it went.
LARGE_AMOUNT = 50_000
# "Sent on quickly" means within this many days of a large credit.
QUICK_EXIT_DAYS = 2
# A burst of payments from many senders is looked for within any span of this many days.
SENDER_WINDOW_DAYS = 7
# After the last payment in the burst, "cashed out" means within this many days.
CASH_OUT_DAYS = 5
# A payment out is "returned" if money comes back within this many days...
RETURN_DAYS = 15
# ...and the payment out was at least this large...
RETURN_MIN = 10_000
# ...and the money back is 90% to 105% of what went out.
RETURN_LOW, RETURN_HIGH = 0.90, 1.05
# Only these channels carry a payment that can be sent back.
RETURNABLE_CHANNELS = ("neft", "rtgs", "imps", "upi")
# At least this many different senders before a burst of payments is worth describing.
MIN_SENDERS = 3
# How many of the biggest payees to list.
TOP_PAYEES = 3

# Risk comes from how large the amount at issue is compared with monthly income.
# These cut-offs were chosen by looking at the training set only: they give each
# pattern a spread of low, medium and high (about 14%, 47% and 39% overall).
MEDIUM_RATIO = 5.0
HIGH_RATIO = 15.0


@dataclass(frozen=True)
class Features:
    """All the numbers an analyst would want, computed from one alert's transactions."""

    occupation: str
    income: int
    account_age_months: int
    # Cash
    cash_count: int
    cash_total: int
    just_under_count: int
    just_under_total: int
    just_under_span_days: int
    just_under_ratio: float
    # Large credits that left quickly
    large_count: int
    large_total: int
    quick_exit_amount: int
    quick_exit_percent: int
    quick_exit_ratio: float
    # Many senders, then a cash-out
    sender_count: int
    sender_total: int
    sender_span_days: int
    cash_out_percent: int
    sender_ratio: float
    # Money that came back
    return_count: int
    return_total: int
    return_percent: int
    return_same_party: int
    return_ratio: float
    # Overall
    top_payees: List[Tuple[str, int]]
    credits_total: int
    debits_total: int
    credits_ratio: float


def ratio_to_income(amount: int, income: int) -> float:
    """How many months of declared income this amount equals, to one decimal."""
    return round(amount / income, 1) if income else 0.0


def percent(part: int, whole: int) -> int:
    """A whole-number percentage, capped at 100."""
    return min(100, round(100 * part / whole)) if whole else 0


def days_between(first, last) -> int:
    """Days from first to last, counting both ends."""
    return (last - first).days + 1


def cash_features(credits: List[Transaction]) -> Dict[str, int]:
    """Cash deposits, and how many sit just under the reporting limit."""
    cash = [c for c in credits if c.channel == "cash"]
    low = JUST_UNDER_LOW * CASH_REPORTING_LIMIT
    high = JUST_UNDER_HIGH * CASH_REPORTING_LIMIT
    just_under = [c for c in cash if low <= c.amount <= high]
    span = days_between(min(c.day for c in just_under), max(c.day for c in just_under)) if just_under else 0
    return {
        "cash_count": len(cash),
        "cash_total": sum(c.amount for c in cash),
        "just_under_count": len(just_under),
        "just_under_total": sum(c.amount for c in just_under),
        "just_under_span_days": span,
    }


def large_credit_features(credits: List[Transaction], external_debits: List[Transaction]) -> Dict[str, int]:
    """Large credits, and how much of each left for other people within two days."""
    large = [c for c in credits if c.amount >= LARGE_AMOUNT]
    sent_on = 0
    for credit in large:
        out = sum(d.amount for d in external_debits if 0 <= (d.day - credit.day).days <= QUICK_EXIT_DAYS)
        sent_on += min(out, credit.amount)
    large_total = sum(c.amount for c in large)
    return {
        "large_count": len(large),
        "large_total": large_total,
        "quick_exit_amount": sent_on,
        "quick_exit_percent": percent(sent_on, large_total),
    }


def busiest_week(paid_in: List[Transaction]) -> List[Transaction]:
    """The payments in the 7-day span that has the most different senders."""
    ordered = sorted(paid_in, key=lambda c: (c.day, c.txn_id))
    best: List[Transaction] = []
    most_senders = 0
    for first in ordered:
        span = [c for c in ordered if 0 <= (c.day - first.day).days < SENDER_WINDOW_DAYS]
        senders = len({c.counterparty for c in span})
        if senders > most_senders:
            best, most_senders = span, senders
    return best


def sender_features(credits: List[Transaction], external_debits: List[Transaction]) -> Dict[str, int]:
    """The busiest burst of UPI or IMPS payments from different senders, and how fast it was cashed out.

    Looking at one burst, not the whole month, keeps regular income (such as weekly
    payouts) from hiding a burst of payments.
    """
    paid_in = [c for c in credits if c.channel in ("upi", "imps")]
    burst = busiest_week(paid_in)
    if not burst:
        return {"sender_count": 0, "sender_total": 0, "sender_span_days": 0, "cash_out_percent": 0}
    last_day = max(c.day for c in burst)
    total = sum(c.amount for c in burst)
    left = sum(d.amount for d in external_debits if 0 <= (d.day - last_day).days <= CASH_OUT_DAYS)
    return {
        "sender_count": len({c.counterparty for c in burst}),
        "sender_total": total,
        "sender_span_days": days_between(min(c.day for c in burst), last_day),
        "cash_out_percent": percent(left, total),
    }


def return_features(credits: List[Transaction], external_debits: List[Transaction]) -> Dict[str, int]:
    """Transfers out that came back, nearly whole, within fifteen days.

    Only transfers can be returned. Cash withdrawals, cash deposits and salary are ignored.
    """
    used = set()
    pairs = []
    sent_out = [d for d in external_debits if d.channel in RETURNABLE_CHANNELS and d.amount >= RETURN_MIN]
    paid_in = [c for c in credits if c.channel in RETURNABLE_CHANNELS]
    for debit in sent_out:
        for credit in paid_in:
            if credit.txn_id in used:
                continue
            gap = (credit.day - debit.day).days
            if 0 < gap <= RETURN_DAYS and RETURN_LOW * debit.amount <= credit.amount <= RETURN_HIGH * debit.amount:
                used.add(credit.txn_id)
                pairs.append((debit, credit))
                break
    sent = sum(d.amount for d, _ in pairs)
    returned = sum(c.amount for _, c in pairs)
    return {
        "return_count": len(pairs),
        "return_total": returned,
        "return_percent": percent(returned, sent),
        "return_same_party": sum(1 for d, c in pairs if d.counterparty == c.counterparty),
    }


def biggest_payees(debits: List[Transaction]) -> List[Tuple[str, int]]:
    """The few counterparties that received the most money."""
    totals: Dict[str, int] = defaultdict(int)
    for debit in debits:
        totals[debit.counterparty] += debit.amount
    ranked = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return ranked[:TOP_PAYEES]


def compute_features(raw: RawAlert) -> Features:
    """Turn one alert's transactions into numbers."""
    credits = [t for t in raw.transactions if t.direction == "credit"]
    debits = [t for t in raw.transactions if t.direction == "debit"]
    # Money moved to the customer's own deposit is not "sent on to other people".
    external_debits = [d for d in debits if d.channel != "internal"]
    income = raw.customer.declared_monthly_income

    cash = cash_features(credits)
    large = large_credit_features(credits, external_debits)
    senders = sender_features(credits, external_debits)
    returns = return_features(credits, external_debits)
    credits_total = sum(c.amount for c in credits)

    return Features(
        occupation=raw.customer.occupation,
        income=income,
        account_age_months=raw.customer.account_age_months,
        just_under_ratio=ratio_to_income(cash["just_under_total"], income),
        quick_exit_ratio=ratio_to_income(large["quick_exit_amount"], income),
        sender_ratio=ratio_to_income(senders["sender_total"], income),
        return_ratio=ratio_to_income(returns["return_total"], income),
        top_payees=biggest_payees(debits),
        credits_total=credits_total,
        debits_total=sum(d.amount for d in debits),
        credits_ratio=ratio_to_income(credits_total, income),
        **cash, **large, **senders, **returns,
    )


# --------------------------------------------------------------------------
# Labels derived from the numbers: risk level and reason
# --------------------------------------------------------------------------

def risk_level(typology: str, features: Features) -> str:
    """low, medium or high, from the amount at issue compared with monthly income."""
    if typology == NO_CONCERN:
        return "low"
    amount_ratio = {
        "STRUCTURING": features.just_under_ratio,
        "RAPID_PASS_THROUGH": features.quick_exit_ratio,
        "MULE_ACCOUNT": features.sender_ratio,
        "ROUND_TRIPPING": features.return_ratio,
    }[typology]
    if amount_ratio >= HIGH_RATIO:
        return "high"
    if amount_ratio >= MEDIUM_RATIO:
        return "medium"
    return "low"


# Reasons for normal activity depend on which look-alike it was.
NO_CONCERN_REASONS = {
    None: "Activity matches the declared {occupation} profile; no laundering pattern found.",
    "cash_heavy_shop": "Frequent cash deposits are steady, well below the limit and fit a {occupation}'s business.",
    "house_sale": "One large credit moved to the customer's own fixed deposit; nothing was sent on to other parties.",
    "salary_then_commitments": "Salary went to regular payees (rent, loan EMI, SIP) in line with income; no pass-through.",
    "repay_friend": "A small payment was returned in full by the same person; not a layering pattern.",
}


def reason_for(raw: RawAlert, features: Features) -> str:
    """One sentence explaining the label. Every number comes from `features`."""
    f = features
    if raw.typology == "STRUCTURING":
        return (f"{plural(f.just_under_count, 'cash deposit')} just under the {format_rupees(CASH_REPORTING_LIMIT)} limit "
                f"within {f.just_under_span_days} days, totalling {format_rupees(f.just_under_total)}.")
    if raw.typology == "RAPID_PASS_THROUGH":
        return (f"{f.quick_exit_percent}% of the {format_rupees(f.large_total)} received in large credits "
                f"was sent on within {QUICK_EXIT_DAYS} days.")
    if raw.typology == "MULE_ACCOUNT":
        return (f"{plural(f.sender_count, 'different sender')} paid {format_rupees(f.sender_total)} into a "
                f"{f.account_age_months}-month-old account, and {f.cash_out_percent}% left within {CASH_OUT_DAYS} days.")
    if raw.typology == "ROUND_TRIPPING":
        all_same = f.return_count > 0 and f.return_same_party == f.return_count
        parties = "the same parties" if all_same else "different parties"
        return (f"{plural(f.return_count, 'outgoing payment')} came back within {RETURN_DAYS} days, "
                f"{format_rupees(f.return_total)} in total, from {parties}.")
    return NO_CONCERN_REASONS[raw.hard_negative_kind].format(occupation=f.occupation)
