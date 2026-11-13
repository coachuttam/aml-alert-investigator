"""The "Live run" tab: upload a transactions file and get a checked verdict from the model.

Kept apart from app.py so each file stays short. app.py passes in the function that
draws one alert, so an uploaded alert looks exactly like a saved one.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import streamlit as st

from src.live import build_live_alert, new_upload_id
from src.models import Customer
from src.remote_model import live_generate
from src.upload import UploadError, read_upload, transaction_rows

SAMPLE_FILE = Path(__file__).resolve().parent / "data" / "sample_upload.csv"
# The profile that goes with the sample file, so the sample works without typing.
SAMPLE_OCCUPATION, SAMPLE_INCOME, SAMPLE_ACCOUNT_AGE = "delivery driver", 31000, 34
LIVE_KEY = "live_result"        # where the last live result is kept between screen refreshes


def show_intro() -> None:
    st.caption("Upload one customer's transactions for a 30-day window. Python computes the numbers, "
               "the fine-tuned model labels the alert, and every answer is checked before it is shown. "
               "Use synthetic or made-up data only.")
    st.download_button("Download a sample file", SAMPLE_FILE.read_text(encoding="utf-8"),
                       file_name="sample_upload.csv", mime="text/csv")
    if live_generate() is None:
        st.warning("The live model is not connected here, so uploads go to \"needs review\". "
                   "The computed findings and the rules still run.")


def ask_profile() -> Customer:
    """The customer profile an analyst would have next to the alert."""
    left, middle, right = st.columns(3)
    occupation = left.text_input("Occupation", SAMPLE_OCCUPATION)
    income = middle.number_input("Declared monthly income (₹)", min_value=0, value=SAMPLE_INCOME, step=1000)
    age = right.number_input("Account age (months)", min_value=0, value=SAMPLE_ACCOUNT_AGE, step=1)
    return Customer("C-upload", occupation.strip() or "not stated", int(income), int(age))


def run_live(csv_text: str, customer: Customer, tuned_config) -> None:
    """Read the file, ask the model and keep the result for the screen."""
    try:
        raw = read_upload(csv_text, new_upload_id(), customer)
    except UploadError as problem:
        st.error(str(problem))
        return
    started = time.monotonic()
    with st.spinner("Asking the fine-tuned model (the first call can take a minute)..."):
        alert = build_live_alert(raw, live_generate(), tuned_config)
    st.session_state[LIVE_KEY] = (alert, transaction_rows(raw), time.monotonic() - started)


def show_upload(tuned_config, show_alert: Callable) -> None:
    """The whole tab: the form, then the result drawn like any other alert."""
    show_intro()
    uploaded = st.file_uploader("Transactions (CSV)", type=["csv"])
    customer = ask_profile()
    if st.button("Investigate", type="primary", disabled=uploaded is None):
        run_live(uploaded.getvalue().decode("utf-8-sig", errors="replace"), customer, tuned_config)
    if LIVE_KEY not in st.session_state:
        return
    alert, transactions, seconds = st.session_state[LIVE_KEY]
    st.caption(f"Answered in {seconds:.1f} seconds.")
    show_alert(alert, transactions)
