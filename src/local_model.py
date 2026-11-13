"""Runs the fine-tuned model on this computer, for a demo with no cloud GPU.

    pip install -r requirements-local.txt
    AML_LOCAL_MODEL=1 streamlit run app.py

The first live run downloads the base model (about 3 GB) and loads it, which takes a few
minutes; after that the model stays in memory and answers in seconds. Nothing leaves the
computer: the alert is read, summarised and classified here.

The published scores came from the 4-bit copy of the base model on a T4 GPU (src/predict.py).
That 4-bit library does not run on a Mac, so this file loads the base model at half
precision instead, with the same adapter and the same prompt. Answers should match
closely, but this exact setup was not the one scored.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Optional

from src.prompt import build_messages

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_REPO = "coachuttam/aml-alert-investigator-qwen2.5-1.5b-qlora"
MAX_NEW_TOKENS = 120                    # the same limit as src/predict.py
LOCAL_SETTING = "AML_LOCAL_MODEL"       # set to 1 to run the model on this computer

GenerateFunction = Callable[[str], str]


def wants_local_model(environment=None) -> bool:
    """True when the app was started with AML_LOCAL_MODEL=1."""
    environment = os.environ if environment is None else environment
    return (environment.get(LOCAL_SETTING) or "").strip().lower() in ("1", "true", "yes")


def pick_device(torch) -> str:
    """The fastest place to run: an Apple GPU, an NVIDIA GPU, or the processor."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def load_local_model(adapter: str = ADAPTER_REPO, base_model: str = BASE_MODEL) -> GenerateFunction:
    """Load the base model and the adapter once; return a function that answers one summary."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(torch)
    # Half precision halves the memory on a GPU; a processor needs full precision.
    precision = torch.float32 if device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model).to(device=device, dtype=precision)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def generate(summary: str) -> str:
        prompt = tokenizer.apply_chat_template(build_messages(summary), tokenize=False,
                                               add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    return generate


def local_generate(environment=None) -> Optional[GenerateFunction]:
    """The on-computer model when it was asked for, or None."""
    return load_local_model() if wants_local_model(environment) else None
