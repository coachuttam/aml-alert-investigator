"""The live run: an uploaded alert goes through the same path as the saved ones.

    uploaded file -> features (Python) -> summary -> fine-tuned model -> checks -> review screen

Nothing here is new logic. It joins the existing steps, so a live answer is computed,
checked and shown exactly like a saved one. If the model cannot be reached, the alert
goes to "needs review" and the rules and computed findings are still shown.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Callable, Optional

from src.features import compute_features
from src.models import RawAlert
from src.remote_model import ModelUnavailable
from src.review import ReviewAlert, build_review_alert
from src.schema import CheckedAnswer, check_answer
from src.templates import pick_template, render_summary
from src.upload import NOT_KNOWN

MODEL_NOT_CONNECTED = "the live model is not connected here"

GenerateFunction = Callable[[str], str]


def new_upload_id(now: Optional[datetime] = None) -> str:
    """An id for an uploaded alert, such as 'U-20261113-173012'."""
    return (now or datetime.now()).strftime("U-%Y%m%d-%H%M%S")


def build_live_example(raw: RawAlert) -> dict:
    """What the model reads for an uploaded alert, in the same form as a dataset example."""
    features = compute_features(raw)
    template_id = pick_template(raw.alert_id, raw.template_group)
    feature_values = asdict(features)
    feature_values["top_payees"] = [list(payee) for payee in features.top_payees]
    return {"alert_id": raw.alert_id, "customer_id": raw.customer.customer_id, "split": raw.split,
            "template_id": template_id, "hard_negative_kind": None,
            "summary": render_summary(template_id, features),
            "label": {"typology": NOT_KNOWN, "risk": "unknown", "reason": ""},
            "features": feature_values}


def failed_answer(problem: str) -> CheckedAnswer:
    """An answer that sends the alert to "needs review", with the reason in plain words."""
    return CheckedAnswer(False, None, None, None, [problem])


def ask_model(summary: str, generate: Optional[GenerateFunction]) -> CheckedAnswer:
    """Ask the live model and check its answer; retry once if the checks fail."""
    if generate is None:
        return failed_answer(MODEL_NOT_CONNECTED)
    try:
        checked = check_answer(generate(summary), summary)
        return checked if checked.valid else check_answer(generate(summary), summary)
    except ModelUnavailable as problem:
        return failed_answer(str(problem))


def build_live_alert(raw: RawAlert, generate: Optional[GenerateFunction], tuned_config) -> ReviewAlert:
    """Run an uploaded alert through features, summary, model and checks."""
    example = build_live_example(raw)
    checked = ask_model(example["summary"], generate)
    answer = {"valid": checked.valid, "typology": checked.typology, "risk": checked.risk,
              "reason": checked.reason, "problems": checked.problems}
    return build_review_alert(example, tuned_config, answer, None, raw.window_end.isoformat())
