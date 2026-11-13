"""Hosts the fine-tuned model on Modal, so the app's "Live run" tab has a GPU to call.

    pip install modal && modal setup          (once, on the laptop)
    modal deploy model_server.py              (prints the address of the endpoint)

The server loads the model with the same code the evaluation used (src/predict.py), so a
live answer comes from the same model, prompt and settings as the published scores.
It sleeps when idle and wakes on the first request, which takes about a minute.
Requests need a Modal proxy auth token (Modal-Key and Modal-Secret headers), so only
the app can call it. Nothing is stored: a summary comes in, the answer text goes out.
"""
from __future__ import annotations

import modal

ADAPTER_REPO = "coachuttam/aml-alert-investigator-qwen2.5-1.5b-qlora"
GPU = "T4"                      # the same GPU type the model was trained and scored on
SLEEP_AFTER_SECONDS = 300       # stay awake for 5 minutes after the last request, then stop billing
MAX_SUMMARY_LENGTH = 6000       # a real summary is far shorter; this refuses anything else
MODEL_CACHE = "/root/.cache/huggingface"

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes", "fastapi[standard]")
         .add_local_python_source("src"))
app = modal.App("aml-alert-investigator", image=image)
# Keeps the downloaded model between wake-ups, so only the first start downloads it.
model_cache = modal.Volume.from_name("aml-alert-investigator-model", create_if_missing=True)


@app.cls(gpu=GPU, volumes={MODEL_CACHE: model_cache}, scaledown_window=SLEEP_AFTER_SECONDS)
class Investigator:
    """One loaded copy of the fine-tuned model."""

    @modal.enter()
    def load(self) -> None:
        from src.predict import load_model
        self.generate, _ = load_model(ADAPTER_REPO)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def classify(self, body: dict) -> dict:
        """{"summary": "..."} in, {"text": "<the model's raw answer>"} out."""
        summary = str(body.get("summary", ""))[:MAX_SUMMARY_LENGTH]
        return {"text": self.generate(summary)}
