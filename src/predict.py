"""Runs a model on alert summaries and checks every answer.

The checking logic (`classify_with_retry`) needs no GPU, so it is tested with a fake
model. `load_model` and `generate_text` need torch and run on Colab or a GPU machine:

    python -m src.predict --adapter /path/to/adapter --split test --output results/finetuned_test.jsonl
    python -m src.predict --split test --output results/base_test.jsonl      (plain base model)

The held-out split may be generated, but must not be scored or read before the final run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, List, Optional

from src.dataset import load_split
from src.prompt import build_messages
from src.schema import CheckedAnswer, check_answer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 120            # the JSON answer is well under this
DATA_FOLDER = Path(__file__).resolve().parent.parent / "data"

# Anything that turns a summary into raw model text. Real models and test fakes both fit.
GenerateFunction = Callable[[str], str]
BatchFunction = Callable[[List[str]], List[str]]
BATCH_SIZE = 16                 # alerts answered at once; fits a T4 comfortably


def classify_with_retry(summary: str, generate: GenerateFunction) -> CheckedAnswer:
    """Ask the model; if the answer fails the checks, ask once more.

    A second failure is returned as-is (valid=False), so the alert goes to "needs review".
    """
    checked = check_answer(generate(summary), summary)
    if checked.valid:
        return checked
    return check_answer(generate(summary), summary)


def load_model(adapter_folder: Optional[str], base_model: str = BASE_MODEL):
    """Load the base model (plus the adapter, if given).

    Returns two functions: one answers a single summary, one answers a list of summaries
    at once, which is much faster on a GPU.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"          # so every answer starts right after its own prompt
    quantisation = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quantisation, device_map={"": 0})
    if adapter_folder:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_folder)
    model.eval()

    def generate_batch(summaries: List[str]) -> List[str]:
        prompts = [tokenizer.apply_chat_template(build_messages(s), tokenize=False,
                                                 add_generation_prompt=True) for s in summaries]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    def generate(summary: str) -> str:
        return generate_batch([summary])[0]

    return generate, generate_batch


def predict_split(examples: List[dict], generate: GenerateFunction,
                  generate_batch: Optional[BatchFunction] = None,
                  batch_size: int = BATCH_SIZE) -> List[dict]:
    """One saved row per alert: the checked answer, ready for scoring.

    With a batch function, alerts are answered a batch at a time; an answer that fails
    the checks is retried once on its own, as in classify_with_retry.
    """
    rows = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        if generate_batch is None:
            checked_answers = [classify_with_retry(e["summary"], generate) for e in chunk]
        else:
            texts = generate_batch([e["summary"] for e in chunk])
            checked_answers = [check_with_one_retry(e["summary"], text, generate)
                               for e, text in zip(chunk, texts)]
        rows.extend(to_row(e, checked) for e, checked in zip(chunk, checked_answers))
        print(f"  answered {len(rows)} of {len(examples)}", flush=True)
    return rows


def check_with_one_retry(summary: str, first_text: str, generate: GenerateFunction) -> CheckedAnswer:
    """Check an answer already generated; if it fails, ask once more on its own."""
    checked = check_answer(first_text, summary)
    return checked if checked.valid else check_answer(generate(summary), summary)


def to_row(example: dict, checked: CheckedAnswer) -> dict:
    """The saved form of one checked answer."""
    return {"alert_id": example["alert_id"], "valid": checked.valid,
            "typology": checked.typology, "risk": checked.risk,
            "reason": checked.reason, "problems": checked.problems}


def save_predictions(rows: List[dict], path: Path) -> None:
    """Write one JSON line per alert."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run a model over one split and save its answers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=None, help="adapter folder; omit to run the plain base model")
    parser.add_argument("--split", default="test", choices=["validation", "test", "heldout"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    examples = load_split(DATA_FOLDER / f"{args.split}.jsonl")[: args.limit]
    generate, generate_batch = load_model(args.adapter)
    save_predictions(predict_split(examples, generate, generate_batch), Path(args.output))
    print(f"Wrote {len(examples)} answers to {args.output}")


if __name__ == "__main__":
    main()
