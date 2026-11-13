"""Calls the fine-tuned model where it is hosted, over HTTPS.

The app has no GPU, so the model runs on a separate machine. This file sends it one
alert summary and gets the raw answer text back:

    POST <AML_MODEL_URL>   {"summary": "..."}   ->   {"text": "{\"typology\": ...}"}

The server builds the prompt with src/prompt.py, so training and live wording stay the same.
The address and token come from the environment (on Streamlit Community Cloud, from the
app's secrets), never from a file in the repo. Only the standard library is used.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Optional

URL_SETTING = "AML_MODEL_URL"
TOKEN_SETTING = "AML_MODEL_TOKEN"
TIMEOUT_SECONDS = 90            # a sleeping GPU machine can take about a minute to wake up

GenerateFunction = Callable[[str], str]


class ModelUnavailable(Exception):
    """Raised when the hosted model cannot be reached or gives an unusable reply."""


def post_summary(url: str, token: str, summary: str, timeout: int = TIMEOUT_SECONDS) -> str:
    """Send one summary and return the model's raw answer text."""
    body = json.dumps({"summary": summary}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            reply = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError) as problem:
        raise ModelUnavailable(f"the live model did not answer ({problem})")
    if not isinstance(reply, dict) or not isinstance(reply.get("text"), str):
        raise ModelUnavailable("the live model's reply had no text")
    return reply["text"]


def live_generate(environment=None) -> Optional[GenerateFunction]:
    """A function that asks the hosted model, or None when no address is set."""
    environment = os.environ if environment is None else environment
    url = (environment.get(URL_SETTING) or "").strip()
    if not url.startswith("https://"):
        return None
    token = (environment.get(TOKEN_SETTING) or "").strip()
    return lambda summary: post_summary(url, token, summary)
