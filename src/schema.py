"""The fixed lists of labels that every part of the project agrees on.

It also checks a model's answer before anyone sees it: valid JSON, known labels,
a short reason, no accusing words, and no number that is not in the alert summary.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

# The four money-laundering patterns the model must recognise.
TYPOLOGIES = (
    "STRUCTURING",         # cash split into deposits just under a reporting limit
    "RAPID_PASS_THROUGH",  # large money in, almost all of it out again within days
    "MULE_ACCOUNT",        # many unrelated senders, then a quick cash-out
    "ROUND_TRIPPING",      # money sent out comes back through other parties
)

# The answer for normal activity, including legitimate look-alikes.
NO_CONCERN = "NO_CONCERN"

ALL_LABELS = TYPOLOGIES + (NO_CONCERN,)

RISK_LEVELS = ("low", "medium", "high")

# Every customer belongs to exactly one of these sets.
SPLITS = ("train", "validation", "test", "heldout")

MAX_REASON_LENGTH = 200
BANNED_WORDS = ("fraud", "criminal", "guilty")   # we refer for review, we never accuse


@dataclass(frozen=True)
class CheckedAnswer:
    """The result of checking one model answer."""
    valid: bool
    typology: Optional[str]
    risk: Optional[str]
    reason: Optional[str]
    problems: List[str]


def numbers_in(text: str) -> set:
    """Every number in the text, with thousands separators removed."""
    return {match.replace(",", "") for match in re.findall(r"\d[\d,]*", text)}


def parse_json_object(raw_text: str) -> Optional[dict]:
    """The JSON object in the text, or None. Tolerates text around the braces."""
    start, end = raw_text.find("{"), raw_text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(raw_text[start:end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def find_problems(answer: dict, summary: str) -> List[str]:
    """Everything wrong with a parsed answer. An empty list means it is fine."""
    problems = []
    if answer.get("typology") not in ALL_LABELS:
        problems.append("typology is not one of the allowed labels")
    if answer.get("risk") not in RISK_LEVELS:
        problems.append("risk is not one of the allowed levels")
    reason = answer.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return problems + ["reason is missing"]
    if len(reason) > MAX_REASON_LENGTH:
        problems.append("reason is too long")
    if any(word in reason.lower() for word in BANNED_WORDS):
        problems.append("reason uses accusing wording")
    if numbers_in(reason) - numbers_in(summary):
        problems.append("reason uses a number that is not in the summary")
    return problems


def check_answer(raw_text: str, summary: str) -> CheckedAnswer:
    """Parse and check the model's raw text against the alert summary it was given."""
    answer = parse_json_object(raw_text)
    if answer is None:
        return CheckedAnswer(False, None, None, None, ["output is not valid JSON"])
    problems = find_problems(answer, summary)
    return CheckedAnswer(not problems, answer.get("typology"), answer.get("risk"),
                         answer.get("reason"), problems)
