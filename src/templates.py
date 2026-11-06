"""The wording used to turn numbers into an alert summary.

Every template shows exactly the same facts; only the words and layout differ.
The first three templates are shared by train, validation and test. The last two
are reserved for the held-out set, so that set tests whether the model learned
the pattern and not the phrasing.

Placeholders such as {cash_total} are filled from the computed features.
Wording is chosen so it reads correctly when a count is 1.
"""
from __future__ import annotations

import random
from typing import Dict, List

from src.features import (CASH_OUT_DAYS, LARGE_AMOUNT, QUICK_EXIT_DAYS, RETURN_DAYS,
                          RETURN_MIN, SENDER_WINDOW_DAYS, Features)
from src.money import format_rupees, plural
from src.patterns import CASH_REPORTING_LIMIT

SHARED_TEMPLATES = ["digest", "prose", "terse"]
HELDOUT_TEMPLATES = ["report", "bullets"]

TEMPLATES: Dict[str, Dict[str, str]] = {
    "digest": {
        "header": "",
        "joiner": "\n",
        "profile": "Customer: {occupation}, declared monthly income {income}, account open {age} months.",
        "cash": "Cash: {cash_deposits} totalling {cash_total}; just under the {limit} limit: {just_under_count} ({just_under_total} in total, {just_under_ratio}x monthly income) over {span} days.",
        "cash_plain": "Cash: {cash_deposits} totalling {cash_total}; none just under the {limit} limit.",
        "cash_none": "Cash: no deposits.",
        "inflows": "Large credits: {large_count} of {large_min} or more, {large_total} in total; {quick_pct}% ({quick_amount}, {quick_ratio}x monthly income) was sent on to others within {quick_days} days.",
        "inflows_none": "Large credits: none of {large_min} or more.",
        "senders": "Senders: {sender_count} different senders paid in {sender_total} ({sender_ratio}x monthly income) over {sender_days} days; {cash_out_pct}% of it left within {cash_out_days} days.",
        "senders_none": "Senders: fewer than 3 different senders in any {sender_window} days.",
        "returns": "Returned money: {return_payments} of {return_min} or more came back within {return_days} days (from the same party: {same_party}), {return_total} in total, {return_pct}% of what was sent ({return_ratio}x monthly income).",
        "returns_none": "Returned money: none.",
        "payees": "Biggest payees: {payees}.",
        "totals": "Total in {credits_total} ({credits_ratio}x monthly income), total out {debits_total}.",
    },
    "prose": {
        "header": "",
        "joiner": " ",
        "profile": "The customer is a {occupation} with a declared monthly income of {income}; the account is {age} months old.",
        "cash": "They made {cash_deposits} worth {cash_total}, of which {just_under_count} sat just under the {limit} limit ({just_under_total}, or {just_under_ratio} times monthly income) within {span} days.",
        "cash_plain": "They made {cash_deposits} worth {cash_total}, none of them just under the {limit} limit.",
        "cash_none": "There were no cash deposits.",
        "inflows": "{large_credits} of {large_min} or more arrived, {large_total} in all, and {quick_pct}% of that ({quick_amount}, or {quick_ratio} times monthly income) was passed on to others within {quick_days} days.",
        "inflows_none": "No single credit reached {large_min}.",
        "senders": "{sender_count} different senders paid in {sender_total} ({sender_ratio} times monthly income) over {sender_days} days, and {cash_out_pct}% of it left within {cash_out_days} days.",
        "senders_none": "Fewer than 3 different senders paid in during any {sender_window} days.",
        "returns": "{return_payments} of {return_min} or more came back within {return_days} days, {same_party} from the same party, {return_total} in all ({return_pct}% of what went out, {return_ratio} times monthly income).",
        "returns_none": "No transfer of {return_min} or more came back.",
        "payees": "The biggest payees were {payees}.",
        "totals": "Total money in was {credits_total} ({credits_ratio} times monthly income) and total out was {debits_total}.",
    },
    "terse": {
        "header": "",
        "joiner": " | ",
        "profile": "{occupation} | income {income}/month | account {age} months",
        "cash": "cash in: count {cash_count}, total {cash_total}; just under {limit}: {just_under_count} ({just_under_total}, {just_under_ratio}x income, {span} days)",
        "cash_plain": "cash in: count {cash_count}, total {cash_total}; just under {limit}: none",
        "cash_none": "cash in: none",
        "inflows": "credits of {large_min}+: {large_count} ({large_total}); sent on within {quick_days} days: {quick_pct}% ({quick_amount}, {quick_ratio}x income)",
        "inflows_none": "credits of {large_min}+: none",
        "senders": "senders: {sender_count} ({sender_total}, {sender_ratio}x income, {sender_days} days); gone within {cash_out_days} days: {cash_out_pct}%",
        "senders_none": "senders: under 3 in any {sender_window} days",
        "returns": "came back within {return_days} days ({return_min}+): count {return_count}, same party {same_party}, {return_total}, {return_pct}% ({return_ratio}x income)",
        "returns_none": "came back: none",
        "payees": "top payees: {payees}",
        "totals": "in {credits_total} ({credits_ratio}x income), out {debits_total}",
    },
    # ---- Reserved for the held-out set: different words, different layout ----
    "report": {
        "header": "ALERT REVIEW",
        "joiner": "\n",
        "profile": "Profile — {occupation}; stated income {income} a month; account age {age} months",
        "cash": "Cash lodgements — count {cash_count}, totalling {cash_total}. Just beneath the {limit} threshold: {just_under_count} ({just_under_total}; {just_under_ratio} times monthly income) across {span} days",
        "cash_plain": "Cash lodgements — count {cash_count}, totalling {cash_total}. None fall just beneath the {limit} threshold",
        "cash_none": "Cash lodgements — none",
        "inflows": "Large receipts — {large_count} of {large_min} or above, total {large_total}. {quick_pct}% ({quick_amount}; {quick_ratio} times monthly income) was remitted onward to third parties within {quick_days} days",
        "inflows_none": "Large receipts — none of {large_min} or above",
        "senders": "Remitters — {sender_count} distinct remitters paid in {sender_total} ({sender_ratio} times monthly income) across {sender_days} days; {cash_out_pct}% had left within {cash_out_days} days",
        "senders_none": "Remitters — fewer than 3 distinct within any {sender_window} days",
        "returns": "Funds returned — {return_payments} of {return_min} or above repaid within {return_days} days (by the same party: {same_party}), total {return_total}; {return_pct}% of the amount sent ({return_ratio} times monthly income)",
        "returns_none": "Funds returned — none",
        "payees": "Main beneficiaries — {payees}",
        "totals": "Turnover — in {credits_total} ({credits_ratio} times monthly income); out {debits_total}",
    },
    "bullets": {
        "header": "Review notes:",
        "joiner": "\n",
        "profile": "- Holder works as: {occupation}. Monthly income on file: {income}. Account has existed for {age} months.",
        "cash": "- Cash paid in: {cash_deposits} totalling {cash_total}. Close to but under the {limit} cut-off: {just_under_count} ({just_under_total}, {just_under_ratio} times monthly income), all within {span} days.",
        "cash_plain": "- Cash paid in: {cash_deposits} totalling {cash_total}. None were close to the {limit} cut-off.",
        "cash_none": "- Cash paid in: never.",
        "inflows": "- Big pay-ins ({large_min} or more): {large_count}, {large_total} altogether. Moved on to other people inside {quick_days} days: {quick_pct}% ({quick_amount}, {quick_ratio} times monthly income).",
        "inflows_none": "- Big pay-ins ({large_min} or more): none.",
        "senders": "- Pay-ins from {sender_count} separate people or firms, {sender_total} altogether ({sender_ratio} times monthly income), spread over {sender_days} days. {cash_out_pct}% had gone out again inside {cash_out_days} days.",
        "senders_none": "- Pay-ins from fewer than 3 separate people or firms in any {sender_window} days.",
        "returns": "- Money that came back: {return_payments} of {return_min} or more returned inside {return_days} days (from the same party: {same_party}), {return_total} altogether, {return_pct}% of what was paid ({return_ratio} times monthly income).",
        "returns_none": "- Money that came back: none.",
        "payees": "- Most money went to: {payees}.",
        "totals": "- Overall: {credits_total} in ({credits_ratio} times monthly income), {debits_total} out.",
    },
}


def template_context(features: Features) -> Dict[str, str]:
    """Every placeholder value, already written out as text."""
    f = features
    payees = ", ".join(f"{name} {format_rupees(amount)}" for name, amount in f.top_payees)
    return {
        "occupation": f.occupation, "income": format_rupees(f.income), "age": str(f.account_age_months),
        "limit": format_rupees(CASH_REPORTING_LIMIT),
        "cash_count": str(f.cash_count), "cash_deposits": plural(f.cash_count, "cash deposit"),
        "cash_total": format_rupees(f.cash_total),
        "just_under_count": str(f.just_under_count), "just_under_total": format_rupees(f.just_under_total),
        "just_under_ratio": str(f.just_under_ratio), "span": str(f.just_under_span_days),
        "large_min": format_rupees(LARGE_AMOUNT), "large_count": str(f.large_count),
        "large_credits": plural(f.large_count, "credit"),
        "large_total": format_rupees(f.large_total), "quick_days": str(QUICK_EXIT_DAYS),
        "quick_pct": str(f.quick_exit_percent), "quick_amount": format_rupees(f.quick_exit_amount),
        "quick_ratio": str(f.quick_exit_ratio),
        "sender_count": str(f.sender_count), "sender_total": format_rupees(f.sender_total),
        "sender_ratio": str(f.sender_ratio), "sender_days": str(f.sender_span_days),
        "sender_window": str(SENDER_WINDOW_DAYS),
        "cash_out_pct": str(f.cash_out_percent), "cash_out_days": str(CASH_OUT_DAYS),
        "return_count": str(f.return_count), "return_payments": plural(f.return_count, "payment"),
        "return_min": format_rupees(RETURN_MIN),
        "return_days": str(RETURN_DAYS), "same_party": str(f.return_same_party),
        "return_total": format_rupees(f.return_total), "return_pct": str(f.return_percent),
        "return_ratio": str(f.return_ratio),
        "payees": payees,
        "credits_total": format_rupees(f.credits_total), "credits_ratio": str(f.credits_ratio),
        "debits_total": format_rupees(f.debits_total),
    }


def section_keys(features: Features) -> List[str]:
    """For each section, which wording applies: full, plain or none."""
    f = features
    cash = "cash" if f.just_under_count else ("cash_plain" if f.cash_count else "cash_none")
    return [
        "profile",
        cash,
        "inflows" if f.large_count else "inflows_none",
        "senders" if f.sender_count >= 3 else "senders_none",
        "returns" if f.return_count else "returns_none",
        "payees",
        "totals",
    ]


def pick_template(alert_id: str, template_group: str) -> str:
    """Choose a template for this alert from the group it is allowed to use."""
    options = HELDOUT_TEMPLATES if template_group == "heldout" else SHARED_TEMPLATES
    return random.Random(alert_id).choice(options)


def render_summary(template_id: str, features: Features) -> str:
    """Write the alert summary from the features, using one template."""
    template = TEMPLATES[template_id]
    context = template_context(features)
    lines = [template[key].format(**context) for key in section_keys(features)]
    body = template["joiner"].join(lines)
    return f"{template['header']}\n{body}" if template["header"] else body
