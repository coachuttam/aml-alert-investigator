"""The prompt the model sees. Training and inference both use this file,
so the wording can never drift apart.
"""
from __future__ import annotations

import json
from typing import Dict, List

from src.schema import ALL_LABELS, RISK_LEVELS

SYSTEM_PROMPT = (
    "You are an anti-money-laundering assistant. Read the alert summary and reply "
    "with JSON only, in this form: "
    '{"typology": ..., "risk": ..., "reason": ...}. '
    f"typology is one of: {', '.join(ALL_LABELS)}. "
    f"risk is one of: {', '.join(RISK_LEVELS)}. "
    "reason is one short sentence that uses only numbers found in the summary."
)


def build_messages(summary: str) -> List[Dict[str, str]]:
    """The conversation up to the point where the model should answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": summary},
    ]


def answer_text(label: Dict[str, str]) -> str:
    """The exact answer the model is trained to produce."""
    ordered = {key: label[key] for key in ("typology", "risk", "reason")}
    return json.dumps(ordered, ensure_ascii=False)
