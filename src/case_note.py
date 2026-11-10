"""Plain-English findings and the draft case note. Python writes both from computed numbers.

No model is called here, so the note can never contain a number that Python did not
compute. The wording check refuses any accusing word before a note is shown or saved.
"""
from __future__ import annotations

from typing import List

from src.money import format_rupees, plural
from src.review import NEEDS_REVIEW, ReviewAlert, agreement_label, words_for
from src.schema import BANNED_WORDS, NO_CONCERN

SYNTHETIC_NOTE = "Synthetic data. Prototype, not certified for compliance."


class GuardrailError(Exception):
    """Raised when text contains a word we never use about a customer."""


def assert_clean_wording(text: str) -> None:
    """Refuse text that accuses anyone. We refer for review; we never say fraud."""
    lowered = text.lower()
    for word in BANNED_WORDS:
        if word in lowered:
            raise GuardrailError(f"The wording check refused the word '{word}'.")


def money_line(features: dict) -> str:
    """Money in and out against the declared income (the profile comparison)."""
    return (f"Money in was {format_rupees(features['credits_total'])}, which is {features['credits_ratio']} times "
            f"the declared monthly income of {format_rupees(features['income'])}; "
            f"money out was {format_rupees(features['debits_total'])}.")


def cash_line(features: dict) -> List[str]:
    """Lines about cash deposits just under the reporting limit."""
    if features["just_under_count"] == 0:
        return []
    return [f"{plural(features['just_under_count'], 'cash deposit')} just under the reporting limit within "
            f"{plural(features['just_under_span_days'], 'day')}, totalling {format_rupees(features['just_under_total'])}."]


def pass_through_line(features: dict) -> List[str]:
    """Lines about large credits that were sent on quickly."""
    if features["large_count"] == 0:
        return []
    return [f"{plural(features['large_count'], 'large credit')} totalling {format_rupees(features['large_total'])}; "
            f"{features['quick_exit_percent']}% of that was sent on within 2 days."]


def sender_line(features: dict) -> List[str]:
    """Lines about many different senders in one week, then cash out."""
    if features["sender_count"] < 3:
        return []
    return [f"{features['sender_count']} different senders paid in {format_rupees(features['sender_total'])} within "
            f"{plural(features['sender_span_days'], 'day')}; {features['cash_out_percent']}% of it was cashed out soon after."]


def return_line(features: dict) -> List[str]:
    """Lines about transfers that came back."""
    if features["return_count"] == 0:
        return []
    return [f"{plural(features['return_count'], 'transfer')} came back, totalling {format_rupees(features['return_total'])}."]


def build_findings(features: dict) -> List[str]:
    """Every computed finding about the alert, as short sentences."""
    lines = [money_line(features)]
    for make_lines in (cash_line, pass_through_line, sender_line, return_line):
        lines.extend(make_lines(features))
    if len(lines) == 1:
        lines.append("None of the four patterns showed up in the computed numbers.")
    return lines


def next_step_for(alert: ReviewAlert) -> str:
    """What the analyst should do, in calm words. Nothing is filed or sent."""
    if alert.model_typology == NEEDS_REVIEW:
        return "Review the transactions by hand: the model's answer could not be validated."
    if alert.model_typology == NO_CONCERN:
        return "No further action suggested. The analyst confirms or overrides."
    return "Refer for review by a senior analyst before any action. Nothing has been filed or sent."


def draft_case_note(alert: ReviewAlert, findings: List[str]) -> str:
    """The draft case note as Markdown. Raises GuardrailError if the wording check fails."""
    lines = [
        f"# Case note: {alert.alert_id}",
        "",
        f"**Customer:** {alert.customer_id}  ·  **Window ends:** {alert.window_end or 'not available'}",
        f"**Model verdict:** {words_for(alert.model_typology)}, risk {alert.model_risk}",
    ]
    if alert.model_reason:
        lines.append(f"**Model reason:** {alert.model_reason}")
    lines += ["", "## Computed findings"] + [f"- {finding}" for finding in findings]
    lines += ["", "## Cross-check", agreement_label(alert), "", "## Suggested next step", next_step_for(alert),
              "", f"_{SYNTHETIC_NOTE}_"]
    note = "\n".join(lines)
    assert_clean_wording(note)
    return note
