"""Builds one customer's 30-day transaction history.

Every alert starts with ordinary everyday activity. Then one pattern is planted
on top: a money-laundering typology, or a legitimate look-alike (a "hard
negative"). Nothing here decides a label; the caller already knows which
pattern it asked for.

All planting functions share one signature so the generator can call them from
a table:  plant_x(ledger, rng, customer, start, end, pool, scale).
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Tuple

from src.models import Customer, Transaction
from src.names import NamePool

# Illustrative per-deposit limit for this demo. Not a regulatory figure.
CASH_REPORTING_LIMIT = 50_000

# "Just under the limit" means 80% to 99% of it.
JUST_UNDER_LOW = 0.80
JUST_UNDER_HIGH = 0.99

# Ordinary cash deposits stay well below the "just under" band.
EVERYDAY_CASH_CAP = 0.70 * CASH_REPORTING_LIMIT

MERCHANTS = ("Grocery store", "Pharmacy", "Fuel station", "Food delivery", "Online shopping", "Mobile recharge")


def round_to(value: float, step: int) -> int:
    """Round to the nearest multiple of `step`, as a whole number."""
    return int(round(value / step)) * step


def day_in(rng: random.Random, first: date, last: date) -> date:
    """A random day from `first` to `last`, both included."""
    return first + timedelta(days=rng.randint(0, (last - first).days))


class Ledger:
    """Collects the transactions of one alert and numbers them in date order."""

    def __init__(self, alert_id: str) -> None:
        self.alert_id = alert_id
        self._rows: List[Tuple[date, int, int, str, str, str]] = []

    def credit(self, day: date, amount: int, channel: str, counterparty: str) -> None:
        """Record money coming in."""
        self._add(day, amount, "credit", channel, counterparty)

    def debit(self, day: date, amount: int, channel: str, counterparty: str) -> None:
        """Record money going out."""
        self._add(day, amount, "debit", channel, counterparty)

    def _add(self, day: date, amount: int, direction: str, channel: str, counterparty: str) -> None:
        # The insertion number keeps the order stable when two rows share a day.
        self._rows.append((day, len(self._rows), int(amount), direction, channel, counterparty))

    def first_day_with_channel(self, channel: str) -> Optional[date]:
        """The earliest day a transaction used this channel, or None."""
        days = [row[0] for row in self._rows if row[4] == channel]
        return min(days) if days else None

    def finish(self) -> List[Transaction]:
        """Return the transactions sorted by date, with ids T001, T002, ..."""
        transactions = []
        for number, row in enumerate(sorted(self._rows), start=1):
            day, _, amount, direction, channel, counterparty = row
            txn_id = f"{self.alert_id}-T{number:03d}"
            transactions.append(Transaction(txn_id, day, amount, direction, channel, counterparty))
        return transactions


# --------------------------------------------------------------------------
# Everyday activity: income and spending that every customer has
# --------------------------------------------------------------------------

def add_salary(ledger, rng, customer, start, end, pool):
    """One payroll credit early in the window."""
    amount = round_to(customer.declared_monthly_income * rng.uniform(0.97, 1.03), 100)
    ledger.credit(day_in(rng, start, start + timedelta(days=9)), amount, "salary", "Employer payroll")


def add_shop_receipts(ledger, rng, customer, start, end, pool):
    """Many small customer payments, by UPI and a little cash."""
    count = rng.randint(10, 16)
    typical = customer.declared_monthly_income / count
    for _ in range(count):
        amount = round_to(typical * rng.uniform(0.6, 1.4), 100)
        if rng.random() < 0.7:
            ledger.credit(day_in(rng, start, end), max(amount, 100), "upi", "Customer receipts")
        else:
            ledger.credit(day_in(rng, start, end), min(max(amount, 100), EVERYDAY_CASH_CAP), "cash", "Cash deposit")


def add_freelance_payments(ledger, rng, customer, start, end, pool):
    """A few payments from different clients."""
    count = rng.randint(2, 5)
    total = customer.declared_monthly_income * rng.uniform(0.8, 1.2)
    for _ in range(count):
        amount = round_to(total / count * rng.uniform(0.7, 1.3), 100)
        ledger.credit(day_in(rng, start, end), max(amount, 100), rng.choice(["neft", "upi"]), pool.business(rng))


def add_family_transfer(ledger, rng, customer, start, end, pool):
    """A monthly allowance or household transfer from family."""
    label = "Family transfer" if customer.occupation == "student" else "Spouse transfer"
    amount = round_to(customer.declared_monthly_income * rng.uniform(0.95, 1.05), 100)
    ledger.credit(day_in(rng, start, start + timedelta(days=9)), amount, "neft", label)


def add_weekly_payouts(ledger, rng, customer, start, end, pool):
    """Four weekly payouts from a delivery platform."""
    for week in range(4):
        amount = round_to(customer.declared_monthly_income / 4 * rng.uniform(0.9, 1.1), 100)
        ledger.credit(start + timedelta(days=week * 7 + rng.randint(0, 6)), amount, "upi", "Delivery platform payout")


INCOME_ADDERS: Dict[str, Callable] = {
    "salaried employee": add_salary,
    "shopkeeper": add_shop_receipts,
    "small trader": add_shop_receipts,
    "freelancer": add_freelance_payments,
    "student": add_family_transfer,
    "homemaker": add_family_transfer,
    "delivery driver": add_weekly_payouts,
}


def add_expenses(ledger, rng, customer, start, end, include_rent=True):
    """Rent, bills, small purchases and a few ATM withdrawals."""
    income = customer.declared_monthly_income
    if include_rent and customer.occupation != "student":
        rent = round_to(income * 0.30 * rng.uniform(0.9, 1.1), 500)
        ledger.debit(day_in(rng, start, start + timedelta(days=11)), rent, "neft", "Landlord")
    ledger.debit(day_in(rng, start, end), rng.randint(600, 3_500), "upi", "Electricity board")
    ledger.debit(day_in(rng, start, end), rng.randint(300, 1_200), "upi", "Mobile and broadband")
    for _ in range(rng.randint(8, 16)):
        amount = round_to(rng.uniform(150, max(400, income * 0.04)), 10)
        ledger.debit(day_in(rng, start, end), max(amount, 100), "upi", rng.choice(MERCHANTS))
    for _ in range(rng.randint(1, 3)):
        ledger.debit(day_in(rng, start, end), round_to(rng.uniform(2_000, 10_000), 500), "atm", "ATM withdrawal")


def add_everyday_activity(ledger, rng, customer, start, end, pool, include_rent=True):
    """Income and spending that every customer has, whatever the label."""
    INCOME_ADDERS[customer.occupation](ledger, rng, customer, start, end, pool)
    add_expenses(ledger, rng, customer, start, end, include_rent)


def add_one_off(ledger, rng, customer, start, end, pool):
    """Maybe add a single ordinary one-off, so normal alerts are not all alike."""
    if rng.random() < 0.5:
        return
    income = customer.declared_monthly_income
    kind = rng.choice(["purchase", "premium", "refund"])
    # Amounts scale with income, so a student never pays a premium bigger than a month's allowance.
    if kind == "purchase":
        ledger.debit(day_in(rng, start, end), max(round_to(income * rng.uniform(0.2, 0.6), 500), 500), "upi", "Electronics store")
    elif kind == "premium":
        ledger.debit(day_in(rng, start, end), max(round_to(income * rng.uniform(0.1, 0.3), 500), 500), "neft", "Insurance premium")
    else:
        ledger.credit(day_in(rng, start, end), round_to(rng.uniform(500, 5_000), 100), "upi", "Merchant refund")


# --------------------------------------------------------------------------
# The four typologies
# --------------------------------------------------------------------------

def plant_structuring(ledger, rng, customer, start, end, pool, scale):
    """Many cash deposits just under the reporting limit, then most of it moved on."""
    count = rng.randint(6, 14)
    span = rng.randint(3, 8)
    first_day = day_in(rng, start, end - timedelta(days=span + 3))
    total = 0
    for _ in range(count):
        amount = round_to(CASH_REPORTING_LIMIT * rng.uniform(JUST_UNDER_LOW, JUST_UNDER_HIGH), 100)
        ledger.credit(first_day + timedelta(days=rng.randint(0, span)), amount, "cash", "Cash deposit")
        total += amount
    if rng.random() < 0.7:
        moved = round_to(total * rng.uniform(0.6, 0.9), 1_000)
        ledger.debit(first_day + timedelta(days=span + rng.randint(1, 3)), moved, "neft", pool.person(rng))


def plant_rapid_pass_through(ledger, rng, customer, start, end, pool, scale):
    """Large transfers in, and 90-99% of each sent out again within one or two days."""
    day = day_in(rng, start, start + timedelta(days=8))
    for _ in range(rng.randint(2, 4)):
        amount = round_to(rng.uniform(150_000, 600_000) * scale, 1_000)
        ledger.credit(day, amount, rng.choice(["neft", "rtgs"]), pool.business(rng))
        out_day = day + timedelta(days=rng.randint(1, 2))
        ledger.debit(out_day, round_to(amount * rng.uniform(0.90, 0.99), 100), rng.choice(["neft", "rtgs"]), pool.business(rng))
        day += timedelta(days=rng.randint(2, 6))


def plant_mule_account(ledger, rng, customer, start, end, pool, scale):
    """Many unrelated senders in a few days, then a quick cash-out."""
    span = rng.randint(2, 5)
    first_day = day_in(rng, start, end - timedelta(days=span + 4))
    total = 0
    for sender in pool.people(rng, rng.randint(10, 25)):
        amount = round_to(rng.uniform(3_000, 25_000) * scale, 500)
        ledger.credit(first_day + timedelta(days=rng.randint(0, span)), amount, rng.choice(["upi", "imps"]), sender)
        total += amount
    cash_out_day = first_day + timedelta(days=span + 1)
    atm_total = 0
    for _ in range(rng.randint(3, 5)):
        ledger.debit(cash_out_day + timedelta(days=rng.randint(0, 1)), 10_000, "atm", "ATM withdrawal")
        atm_total += 10_000
    remaining = round_to(total * rng.uniform(0.90, 0.98), 1_000) - atm_total
    if remaining > 0:
        ledger.debit(cash_out_day + timedelta(days=2), remaining, "neft", pool.person(rng))


def plant_round_tripping(ledger, rng, customer, start, end, pool, scale):
    """Money sent to one business comes back, almost whole, from another."""
    day = day_in(rng, start, start + timedelta(days=4))
    for _ in range(rng.randint(2, 3)):
        amount = round_to(rng.uniform(100_000, 400_000) * scale, 1_000)
        ledger.debit(day, amount, "neft", pool.business(rng))
        back_day = day + timedelta(days=rng.randint(3, 6))
        ledger.credit(back_day, round_to(amount * rng.uniform(0.95, 1.0), 100), "neft", pool.business(rng))
        day = back_day + timedelta(days=rng.randint(1, 2))


TYPOLOGY_PLANTERS: Dict[str, Callable] = {
    "STRUCTURING": plant_structuring,
    "RAPID_PASS_THROUGH": plant_rapid_pass_through,
    "MULE_ACCOUNT": plant_mule_account,
    "ROUND_TRIPPING": plant_round_tripping,
}


# --------------------------------------------------------------------------
# Hard negatives: legitimate activity that looks suspicious at first glance
# --------------------------------------------------------------------------

def plant_cash_heavy_shop(ledger, rng, customer, start, end, pool, scale):
    """A real cash business: steady, varied deposits well under the limit."""
    total = 0
    for _ in range(rng.randint(15, 24)):
        amount = round_to(CASH_REPORTING_LIMIT * rng.uniform(0.10, 0.60), 100)
        ledger.credit(day_in(rng, start, end), amount, "cash", "Cash deposit")
        total += amount
    for _ in range(rng.randint(3, 6)):
        ledger.debit(day_in(rng, start, end), round_to(total * rng.uniform(0.08, 0.20), 1_000), "neft", pool.business(rng))


def plant_house_sale(ledger, rng, customer, start, end, pool, scale):
    """One large sale payout, partly put into the customer's own fixed deposit."""
    day = day_in(rng, start + timedelta(days=2), start + timedelta(days=10))
    amount = round_to(rng.uniform(2_000_000, 6_000_000) * scale, 10_000)
    ledger.credit(day, amount, "rtgs", pool.person(rng))
    deposit = round_to(amount * rng.uniform(0.60, 0.80), 10_000)
    ledger.debit(day + timedelta(days=rng.randint(3, 7)), deposit, "internal", "Own fixed deposit")


def plant_salary_then_commitments(ledger, rng, customer, start, end, pool, scale):
    """Salary in, then rent, loan EMI and a mutual-fund SIP out within two days."""
    salary_day = ledger.first_day_with_channel("salary") or start
    income = customer.declared_monthly_income
    commitments = [("Landlord", 0.35), ("Home loan EMI", 0.30), ("Mutual fund SIP", 0.15)]
    for name, share in commitments:
        amount = round_to(income * share * rng.uniform(0.95, 1.05), 500)
        ledger.debit(salary_day + timedelta(days=rng.randint(1, 2)), amount, "neft", name)


def plant_repay_friend(ledger, rng, customer, start, end, pool, scale):
    """A small loan to a friend that is paid back in full."""
    friend = pool.person(rng)
    amount = round_to(rng.uniform(15_000, 40_000) * scale, 1_000)
    lend_day = day_in(rng, start, start + timedelta(days=11))
    ledger.debit(lend_day, amount, "upi", friend)
    ledger.credit(lend_day + timedelta(days=rng.randint(5, 12)), amount, "upi", friend)


HARD_NEGATIVE_PLANTERS: Dict[str, Callable] = {
    "cash_heavy_shop": plant_cash_heavy_shop,
    "house_sale": plant_house_sale,
    "salary_then_commitments": plant_salary_then_commitments,
    "repay_friend": plant_repay_friend,
}
