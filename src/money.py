"""Small text helpers: rupee amounts the way an Indian analyst reads them, and plurals."""
from __future__ import annotations


def plural(count: int, noun: str) -> str:
    """'1 deposit' or '4 deposits'."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def format_rupees(amount: int) -> str:
    """Format with Indian digit grouping: 102000 becomes '₹1,02,000'."""
    digits = str(abs(int(amount)))
    if len(digits) > 3:
        head, tail = digits[:-3], digits[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        digits = ",".join(groups + [tail])
    sign = "-" if amount < 0 else ""
    return f"{sign}₹{digits}"
