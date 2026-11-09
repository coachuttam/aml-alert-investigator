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


def classify_with_retry(summary: str, generate: GenerateFunction) -> CheckedAnswer:
    """Ask the model; if the answer fails the checks, ask once more.

    A second failure is returned as-is (valid=False), so the alert goes to "needs review".
    """
    checked = check_answer(generate(summary), summary)
    if checked.valid:
        return checked
    return check_answer(generate(summary), summary)


def load_model(adapter_folder: Optional[str], base_model: str = BASE_MODEL) -> GenerateFunction:
    """Load the base model (plus the adapter, if given) and return a generate function."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    quantisation = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quantisation, device_map={"": 0})
    if adapter_folder:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_folder)
    model.eval()

    def generate(summary: str) -> str:
        prompt = tokenizer.apply_chat_template(
            build_messages(summary), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return generate


def predict_split(examples: List[dict], generate: GenerateFunction) -> List[dict]:
    """One saved row per alert: the checked answer, ready for scoring."""
    rows = []
    for example in examples:
        checked = classify_with_retry(example["summary"], generate)
        rows.append({"alert_id": example["alert_id"], "valid": checked.valid,
                     "typology": checked.typology, "risk": checked.risk,
                     "reason": checked.reason, "problems": checked.problems})
    return rows


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
    save_predictions(predict_split(examples, load_model(args.adapter)), Path(args.output))
    print(f"Wrote {len(examples)} answers to {args.output}")


if __name__ == "__main__":
    main()
