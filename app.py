"""Review screen for the AML Alert Investigator. Run with:  streamlit run app.py

It shows saved answers from the fine-tuned model next to the rules and the base model,
so no GPU is needed. Layout and wording follow DESIGN_SYSTEM.md.
"""
from __future__ import annotations

from typing import List

import pandas as pd
import streamlit as st

from src.money import format_rupees
from src.review import (HELDOUT_NOTE, NEEDS_REVIEW, ReviewAlert, agreement_label,
                        load_review_alerts, read_decisions, read_transactions, save_decision,
                        supports_verdict, tuned_rules_config, words_for)
from src.review_scores import build_score_table
from src.schema import ALL_LABELS, RISK_LEVELS

SPLIT_CHOICES = {"validation": "validation", "test": "test", "heldout": HELDOUT_NOTE}
RISK_COLOURS = {"low": "rgba(46,160,67,0.18)", "medium": "rgba(210,153,34,0.22)",
                "high": "rgba(218,54,51,0.20)", "unknown": "rgba(128,128,128,0.18)"}
SYNTHETIC_NOTE = "Synthetic data. Prototype, not certified for compliance."

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


def show_table(table: pd.DataFrame, **options) -> None:
    """A full-width table. Newer Streamlit says width="stretch"; older versions say use_container_width."""
    try:
        st.dataframe(table, width="stretch", hide_index=True, **options)
    except (TypeError, ValueError):
        st.dataframe(table, use_container_width=True, hide_index=True, **options)


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


def show_queue(alerts: List[ReviewAlert]) -> None:
    rows = [{"Alert": a.alert_id, "Window end": a.window_end, "Verdict": words_for(a.model_typology),
             "Risk": a.model_risk, "Tuned rules": words_for(a.rules_tuned),
             "Agree": "yes" if a.model_typology == a.rules_tuned else "NO"} for a in alerts]
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
    table = pd.DataFrame({
        "System": ["Fine-tuned model", f"Rules, tuned on train (rule {alert.rules_tuned_rule_id})",
                   "Rules, default thresholds", "Base model (no training)", "True label (synthetic)"],
        "Says": [words_for(alert.model_typology), words_for(alert.rules_tuned),
                 words_for(alert.rules_default), words_for(alert.base_typology), words_for(alert.true_typology)],
    })
    if alert.look_alike_kind:
        st.caption(f"This is a legitimate look-alike alert ({alert.look_alike_kind.replace('_', ' ')}).")
    show_table(table)


def show_evidence(alert: ReviewAlert, transactions: List[dict]) -> None:
    if not transactions:
        st.caption("Transactions are not available here (the raw file is generated by `python -m src.generator`).")
        return
    rows = []
    for txn in sorted(transactions, key=lambda t: t["day"]):
        sign = "+" if txn["direction"] == "credit" else "−"
        rows.append({"Evidence": "●" if supports_verdict(txn, alert.model_typology) else "",
                     "Date": txn["day"], "Channel": txn["channel"],
                     "Amount": f"{sign}{format_rupees(txn['amount'])}", "Counterparty": txn["counterparty"]})
    st.caption("● marks rows behind a structuring or pass-through verdict. Mule and round-trip patterns "
               "need several rows together, so those rows are not marked.")
    show_table(pd.DataFrame(rows), height=320)


def show_decision_panel(alert: ReviewAlert) -> None:
    st.markdown("**Analyst decision**")
    st.caption(f"Model said: {words_for(alert.model_typology)}, risk {alert.model_risk}. "
               "The original alert and verdict are never changed; your decision is saved beside them.")
    action = st.radio("Decision", ["Approve", "Override"], horizontal=True, key=f"action_{alert.alert_id}")
    typology, risk = alert.model_typology, alert.model_risk
    if action == "Override":
        typology = st.selectbox("Typology", list(ALL_LABELS), format_func=words_for, key=f"t_{alert.alert_id}")
        risk = st.selectbox("Risk", list(RISK_LEVELS), key=f"r_{alert.alert_id}")
    note = st.text_input("Note (required for an override)", key=f"n_{alert.alert_id}")
    if st.button("Save decision", key=f"save_{alert.alert_id}"):
        if action == "Override" and not note.strip():
            st.error("Add a note to explain the override.")
        else:
            save_decision(alert, action.lower(), typology, risk, note)
            st.success(f"Saved. Analyst set: {words_for(typology)}, risk {risk}. Nothing was filed or sent.")


def show_detail(alerts: List[ReviewAlert], split: str) -> None:
    if not alerts:
        st.info("No alerts match the filters.")
        return
    chosen_id = st.selectbox("Alert", [alert.alert_id for alert in alerts])
    alert = next(a for a in alerts if a.alert_id == chosen_id)
    transactions = get_transactions(split).get(alert.alert_id, [])
    show_header(alert, len(transactions))
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**What the model read**")
        st.text(alert.summary)
        show_evidence(alert, transactions)
    with right:
        show_verdict(alert)
        show_comparison(alert)
        show_decision_panel(alert)


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


def show_audit_log(decisions: List[dict]) -> None:
    if not decisions:
        st.info("No decisions saved yet. Approve or override an alert and it appears here.")
        return
    table = pd.DataFrame(decisions).rename(columns={"time": "Time", "alert_id": "Alert", "action": "Action",
                                                    "model_said": "Model said", "analyst_set": "Analyst set",
                                                    "note": "Note"})
    show_table(table.iloc[::-1])


def main() -> None:
    st.title("AML Alert Investigator")
    st.caption(SYNTHETIC_NOTE)
    split = st.sidebar.selectbox("Alerts", list(SPLIT_CHOICES), index=1, format_func=SPLIT_CHOICES.get)
    alerts = pick_filters(get_alerts(split))
    st.sidebar.caption(SYNTHETIC_NOTE)
    decisions = read_decisions()
    queue_tab, detail_tab, evaluation_tab, audit_tab = st.tabs(
        [f"Alerts ({len(alerts)})", "Alert detail", "Evaluation", f"Audit log ({len(decisions)})"])
    with queue_tab:
        show_queue(alerts)
    with detail_tab:
        show_detail(alerts, split)
    with evaluation_tab:
        show_evaluation()
    with audit_tab:
        show_audit_log(decisions)


main()
