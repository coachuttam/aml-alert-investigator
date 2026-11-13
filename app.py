"""Review screen for the AML Alert Investigator. Run with:  streamlit run app.py

It shows saved answers from the fine-tuned model next to the rules and the base model,
so no GPU is needed. The "Live run" tab sends an uploaded file to the hosted model. Layout and wording follow DESIGN_SYSTEM.md.
"""
from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from src.audit import read_events
from src.case_store import DECIDED, Case, load_case
from src.evidence import findings_for_rows
from src.investigator import (DecisionError, export_json, export_markdown, record_decision,
                              run_investigation)
from src.money import format_rupees
from src.review import (HELDOUT_NOTE, NEEDS_REVIEW, ReviewAlert, agreement_label,
                        is_shared_demo, load_review_alerts, read_transactions,
                        tuned_rules_config, words_for)
from src.review_scores import build_score_table
from src.schema import ALL_LABELS, RISK_LEVELS
from src.upload import UPLOAD_SPLIT
from evidence_view import show_evidence, show_findings, show_table
from upload_view import show_upload

SPLIT_CHOICES = {"validation": "validation", "test": "test", "heldout": HELDOUT_NOTE}
RISK_COLOURS = {"low": "rgba(46,160,67,0.18)", "medium": "rgba(210,153,34,0.22)",
                "high": "rgba(218,54,51,0.20)", "unknown": "rgba(128,128,128,0.18)"}
SYNTHETIC_NOTE = "Synthetic data. Prototype, not certified for compliance."
STEP_MARKS = {"done": "✔ done", "needs_review": "⚠ needs review", "unavailable": "– not available",
              "waiting": "… waiting for the analyst"}
CASE_WORDS = {"awaiting_analyst": "awaiting analyst", "needs_review": "needs review", "decided": "decided"}

st.set_page_config(page_title="AML Alert Investigator", layout="wide")


@st.cache_data(show_spinner="Loading alerts...")
def get_alerts(split: str) -> List[ReviewAlert]:
    return load_review_alerts(split, get_tuned_config())


@st.cache_data(show_spinner="Tuning the rules on the training set (about 5 seconds)...")
def get_tuned_config():
    return tuned_rules_config()


@st.cache_data
def get_transactions(split: str) -> dict:
    return read_transactions(split)


@st.cache_data(show_spinner="Scoring...")
def get_scores() -> dict:
    return build_score_table(get_tuned_config())


def risk_badge(risk: str) -> str:
    style = (f"background:{RISK_COLOURS.get(risk, RISK_COLOURS['unknown'])};padding:2px 10px;"
             "border-radius:999px;font-size:0.7rem;font-weight:600;")
    return f'<span style="{style}">{risk}</span>'


def verdict_chip(typology: str, source: str) -> str:
    return (f'<span style="font-weight:600;font-size:1.05rem;">{words_for(typology)}</span> '
            f'<span style="opacity:0.65;font-size:0.75rem;">{source}</span>')


def kpi_card(label: str, value: str, sub: str) -> None:
    st.markdown(
        '<div style="border:1px solid rgba(128,128,128,0.25);border-radius:10px;padding:14px;'
        f'min-height:132px;"><div style="font-size:0.78rem;opacity:0.7;">{label}</div>'
        f'<div style="font-size:1.45rem;font-weight:650;">{value}</div>'
        f'<div style="font-size:0.8rem;opacity:0.8;">{sub}</div></div>', unsafe_allow_html=True)


def pick_filters(alerts: List[ReviewAlert]) -> List[ReviewAlert]:
    """Sidebar filters on the model's risk and typology."""
    risks = st.sidebar.multiselect("Risk (model)", list(RISK_LEVELS) + ["unknown"])
    typologies = st.sidebar.multiselect("Verdict (model)", list(ALL_LABELS) + [NEEDS_REVIEW],
                                        format_func=words_for)
    return [alert for alert in alerts
            if (not risks or alert.model_risk in risks)
            and (not typologies or alert.model_typology in typologies)]


def case_status_words(alert_id: str) -> str:
    case = load_case(alert_id)
    return CASE_WORDS[case.status] if case else "not investigated"


def show_queue(alerts: List[ReviewAlert]) -> None:
    rows = [{"Alert": a.alert_id, "Window end": a.window_end, "Verdict": words_for(a.model_typology),
             "Risk": a.model_risk, "Tuned rules": words_for(a.rules_tuned),
             "Agree": "yes" if a.model_typology == a.rules_tuned else "NO",
             "Case": case_status_words(a.alert_id)} for a in alerts]
    st.caption("Highest risk first, then newest. Open an alert in the next tab.")
    show_table(pd.DataFrame(rows))


def show_header(alert: ReviewAlert, transaction_count: int) -> None:
    profile = alert.features
    st.subheader(f"{alert.alert_id}  ·  {alert.customer_id}")
    count_text = f"  ·  {transaction_count} transactions" if transaction_count else ""
    st.caption(f"{profile['occupation']}, declared income {format_rupees(profile['income'])} a month, "
               f"account {profile['account_age_months']} months old  ·  window ends {alert.window_end}{count_text}")
    if alert.model_typology == NEEDS_REVIEW:
        st.warning("Needs review: the model's answer could not be validated. "
                   + "; ".join(alert.model_problems))


def show_verdict(alert: ReviewAlert) -> None:
    st.markdown(verdict_chip(alert.model_typology, "Fine-tuned model") + "  "
                + risk_badge(alert.model_risk), unsafe_allow_html=True)
    if alert.model_reason:
        st.write(alert.model_reason)
        st.caption("✔ Valid JSON  ·  ✔ numbers checked against the input  ·  ✔ no accusing words")
    st.info(agreement_label(alert))


def show_comparison(alert: ReviewAlert) -> None:
    rows = [("Fine-tuned model", alert.model_typology),
            (f"Rules, tuned on train (rule {alert.rules_tuned_rule_id})", alert.rules_tuned),
            ("Rules, default thresholds", alert.rules_default)]
    if alert.split != UPLOAD_SPLIT:      # an uploaded alert has no saved base answer and no true label
        rows += [("Base model (no training)", alert.base_typology), ("True label (synthetic)", alert.true_typology)]
    table = pd.DataFrame({"System": [name for name, _ in rows], "Says": [words_for(label) for _, label in rows]})
    if alert.look_alike_kind:
        st.caption(f"This is a legitimate look-alike alert ({alert.look_alike_kind.replace('_', ' ')}).")
    show_table(table)


def show_steps(case: Case) -> None:
    for step in case.steps:
        with st.expander(f"Step {step.number}: {step.name}  ·  {STEP_MARKS[step.status]}"):
            st.write(step.summary)
            for line in step.details:
                st.markdown(f"- {line}")


def show_decision_panel(alert: ReviewAlert, case: Case) -> None:
    st.markdown("**Step 6: analyst decision**")
    st.caption(f"Model said: {words_for(alert.model_typology)}, risk {alert.model_risk}. "
               "The original alert and verdict are never changed; your decision is saved beside them.")
    if case.decision:
        d = case.decision
        st.success(f"Latest decision: {d['action']}, {words_for(d['typology'])}, risk {d['risk']} ({d['time']}).")
    action = st.radio("Decision", ["Approve", "Override"], horizontal=True, key=f"action_{alert.alert_id}")
    typology, risk = alert.model_typology, alert.model_risk
    if action == "Override" or alert.model_typology == NEEDS_REVIEW:
        start_typology = ALL_LABELS.index(alert.model_typology) if alert.model_typology in ALL_LABELS else 0
        start_risk = RISK_LEVELS.index(alert.model_risk) if alert.model_risk in RISK_LEVELS else 0
        typology = st.selectbox("Typology", list(ALL_LABELS), index=start_typology, format_func=words_for,
                                key=f"t_{alert.alert_id}")
        risk = st.selectbox("Risk", list(RISK_LEVELS), index=start_risk, key=f"r_{alert.alert_id}")
    note = st.text_input("Note (required for an override)", key=f"n_{alert.alert_id}")
    if st.button("Save decision", key=f"save_{alert.alert_id}"):
        try:
            record_decision(alert, case, action.lower(), typology, risk, note)
            st.rerun()
        except DecisionError as problem:
            st.error(str(problem))


def show_exports(case: Case) -> None:
    if case.status != DECIDED:
        st.caption("Export unlocks after the analyst records a decision. Nothing is filed or sent.")
        return
    left, right = st.columns(2)
    left.download_button("Download case note (Markdown)", export_markdown(case),
                         file_name=f"{case.alert_id}.md", mime="text/markdown")
    right.download_button("Download case file (JSON)", export_json(case),
                          file_name=f"{case.alert_id}.json", mime="application/json")


def show_investigation(alert: ReviewAlert, transactions: List[dict]) -> None:
    st.markdown("### Investigation")
    case = load_case(alert.alert_id)
    if case is None:
        st.caption("Six fixed steps: load, fetch transactions, compute findings, classify, draft a note, "
                   "analyst decision. Steps 1 to 5 only read and write local files.")
        if st.button("Run investigation", key=f"run_{alert.alert_id}"):
            run_investigation(alert, transactions)
            st.rerun()
        return
    st.caption(f"Case status: {CASE_WORDS[case.status]}")
    left, right = st.columns([3, 2])
    with left:
        show_steps(case)
        with st.expander("Draft case note", expanded=True):
            st.markdown(case.note)
    with right:
        show_decision_panel(alert, case)
        show_exports(case)


def show_detail(alerts: List[ReviewAlert], split: str) -> None:
    if not alerts:
        st.info("No alerts match the filters.")
        return
    chosen_id = st.selectbox("Alert", [alert.alert_id for alert in alerts])
    alert = next(a for a in alerts if a.alert_id == chosen_id)
    show_alert(alert, get_transactions(split).get(alert.alert_id, []))


def show_alert(alert: ReviewAlert, transactions: List[dict]) -> None:
    """One alert in full: what the model read, the evidence, the verdict and the investigation."""
    show_header(alert, len(transactions))
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**What the model read**")
        st.text(alert.summary)
        findings = findings_for_rows(alert.features, transactions)
        show_findings(findings, transactions)
        show_evidence(findings, transactions)
    with right:
        show_verdict(alert)
        show_comparison(alert)
    show_investigation(alert, transactions)


def show_evaluation() -> None:
    scores = get_scores()
    tuned, finetuned = scores["A2 rules (tuned on train)"], scores["C fine-tuned"]
    columns = st.columns(3)
    with columns[0]:
        kpi_card("Macro-F1, fine-tuned", f"{finetuned['test']['macro_f1']:.3f} / {finetuned['heldout']['macro_f1']:.3f}",
                 f"test / held-out. Tuned rules: {tuned['test']['macro_f1']:.3f} / {tuned['heldout']['macro_f1']:.3f}")
    with columns[1]:
        kpi_card("False alarms on normal alerts", f"{finetuned['test']['false_alarms']:.1%} / {finetuned['heldout']['false_alarms']:.1%}",
                 "fine-tuned, test / held-out")
    with columns[2]:
        kpi_card("Valid JSON answers", f"{finetuned['test']['valid_answers']:.1%} / {finetuned['heldout']['valid_answers']:.1%}",
                 "fine-tuned, test / held-out")
    st.write("")
    rows = []
    for system, by_split in scores.items():
        row = {"System": system}
        for split, label in (("test", "test"), ("heldout", HELDOUT_NOTE)):
            if split in by_split:
                row[f"Macro-F1, {label}"] = f"{by_split[split]['macro_f1']:.3f}"
                row[f"False alarms, {label}"] = f"{by_split[split]['false_alarms']:.1%}"
        rows.append(row)
    show_table(pd.DataFrame(rows))
    st.caption("On held-out wording the fine-tuned model did not beat the tuned rules. "
               "That result is reported as it is; the held-out set was scored once.")


def show_audit_log() -> None:
    events = read_events()
    if not events:
        st.info("Nothing logged yet. Run an investigation and it appears here, step by step.")
        return
    st.caption("Every step and decision, newest first. The log is only ever added to.")
    table = pd.DataFrame([vars(event) for event in events]).rename(columns={
        "time": "Time", "alert_id": "Alert", "action": "Action", "before": "Before", "after": "After", "note": "Note"})
    show_table(table.iloc[::-1])


def main() -> None:
    st.title("AML Alert Investigator")
    st.caption(SYNTHETIC_NOTE)
    if is_shared_demo():
        st.info("Shared demo: saved decisions are visible to other visitors and reset when the app "
                "restarts. Please do not type personal information.")
    split = st.sidebar.selectbox("Alerts", list(SPLIT_CHOICES), index=1, format_func=SPLIT_CHOICES.get)
    alerts = pick_filters(get_alerts(split))
    st.sidebar.caption(SYNTHETIC_NOTE)
    event_count = len(read_events())
    queue_tab, detail_tab, live_tab, evaluation_tab, audit_tab = st.tabs(
        [f"Alerts ({len(alerts)})", "Alert detail", "Live run", "Evaluation", f"Audit log ({event_count})"])
    with queue_tab:
        show_queue(alerts)
    with detail_tab:
        show_detail(alerts, split)
    with live_tab:
        show_upload(get_tuned_config(), show_alert)
    with evaluation_tab:
        show_evaluation()
    with audit_tab:
        show_audit_log()


main()
