"""Reads an uploaded transactions file into a RawAlert, so the live run can use it.

The file is a CSV with one row per transaction and these columns:

    day,amount,direction,channel,counterparty
    2026-09-01,48000,credit,cash,Cash deposit

A `txn_id` column is optional. The customer's profile (occupation, income, account age)
is typed into the app, not read from the file. A bad file raises UploadError with a
plain message that names the row, so the analyst can fix it.
"""
from __future__ import annotations

import csv
import io
from datetime import date
from typing import List

from src.models import Customer, RawAlert, Transaction

REQUIRED_COLUMNS = ("day", "amount", "direction", "channel", "counterparty")
DIRECTIONS = ("credit", "debit")
CHANNELS = ("cash", "upi", "imps", "neft", "rtgs", "atm", "salary", "internal")
MAX_ROWS = 2000                 # a 30-day window is far smaller; this stops a huge file freezing the app
NOT_KNOWN = "NOT_KNOWN"         # an uploaded alert has no true label
UPLOAD_SPLIT = "upload"


class UploadError(Exception):
    """Raised when the uploaded file cannot be used. The message is shown to the analyst."""


def read_amount(text: str, row_number: int) -> int:
    """Whole rupees, positive. '48,000' is accepted."""
    cleaned = text.replace(",", "").strip()
    if not cleaned.isdigit() or int(cleaned) == 0:
        raise UploadError(f"Row {row_number}: amount must be a positive whole number, got '{text}'.")
    return int(cleaned)


def read_day(text: str, row_number: int) -> date:
    """A date written as YYYY-MM-DD."""
    try:
        return date.fromisoformat(text.strip())
    except ValueError:
        raise UploadError(f"Row {row_number}: day must look like 2026-09-01, got '{text}'.")


def read_choice(text: str, allowed: tuple, name: str, row_number: int) -> str:
    """A lower-case value from a fixed list, such as the direction or the channel."""
    value = text.strip().lower()
    if value not in allowed:
        raise UploadError(f"Row {row_number}: {name} must be one of {', '.join(allowed)}; got '{text}'.")
    return value


def read_transaction(row: dict, row_number: int, alert_id: str) -> Transaction:
    """One CSV row as a Transaction. Row numbers start at 2, the first line under the header."""
    counterparty = (row["counterparty"] or "").strip()
    if not counterparty:
        raise UploadError(f"Row {row_number}: counterparty is empty.")
    txn_id = (row.get("txn_id") or "").strip() or f"{alert_id}-T{row_number - 1:03d}"
    return Transaction(txn_id, read_day(row["day"] or "", row_number),
                       read_amount(row["amount"] or "", row_number),
                       read_choice(row["direction"] or "", DIRECTIONS, "direction", row_number),
                       read_choice(row["channel"] or "", CHANNELS, "channel", row_number), counterparty)


def read_transactions(csv_text: str, alert_id: str) -> List[Transaction]:
    """Every row of the file as a Transaction, oldest first."""
    reader = csv.DictReader(io.StringIO(csv_text))
    columns = [name.strip().lower() for name in (reader.fieldnames or [])]
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    if missing:
        raise UploadError(f"The file is missing these columns: {', '.join(missing)}.")
    reader.fieldnames = columns
    rows = list(reader)
    if not rows:
        raise UploadError("The file has no transactions.")
    if len(rows) > MAX_ROWS:
        raise UploadError(f"The file has {len(rows)} rows; the limit is {MAX_ROWS}.")
    transactions = [read_transaction(row, number, alert_id) for number, row in enumerate(rows, start=2)]
    return sorted(transactions, key=lambda txn: txn.day)


def read_upload(csv_text: str, alert_id: str, customer: Customer) -> RawAlert:
    """The uploaded file plus the typed profile, in the same shape the generator makes."""
    if customer.declared_monthly_income <= 0:
        raise UploadError("Declared monthly income must be more than zero.")
    transactions = read_transactions(csv_text, alert_id)
    return RawAlert(alert_id, UPLOAD_SPLIT, "shared", customer, transactions[0].day,
                    transactions[-1].day, transactions, NOT_KNOWN)


def transaction_rows(raw: RawAlert) -> List[dict]:
    """The transactions as plain dicts with text dates, which is what the evidence table reads."""
    return [{"txn_id": t.txn_id, "day": t.day.isoformat(), "amount": t.amount, "direction": t.direction,
             "channel": t.channel, "counterparty": t.counterparty} for t in raw.transactions]
