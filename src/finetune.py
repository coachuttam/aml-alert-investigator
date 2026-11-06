"""QLoRA fine-tuning. Runs on a GPU (Google Colab T4), not on the laptop.

    python -m src.finetune --output-dir /content/drive/MyDrive/aml-adapter

Only the train split is used for learning; the validation split is used to watch
the loss. The test and held-out splits are never read here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from src.dataset import load_split
from src.prompt import answer_text, build_messages

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LENGTH = 512          # summaries are under ~190 tokens, so this is generous
IGNORE_INDEX = -100           # tells the loss to skip these positions
LORA_RANK = 16
LORA_ALPHA = 32               # usually twice the rank
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
EPOCHS = 2
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4        # effective batch of 16 without needing more memory
SAVE_EVERY_STEPS = 100        # checkpoints go to Drive so a disconnect loses little
DATA_FOLDER = Path(__file__).resolve().parent.parent / "data"


def tokenize_example(example: dict, tokenizer, max_length: int = MAX_SEQ_LENGTH) -> Dict[str, List[int]]:
    """Token ids for one example, with the loss switched on for the answer only.

    The prompt part gets IGNORE_INDEX, so the model learns to answer, not to repeat the question.
    """
    prompt_text = tokenizer.apply_chat_template(
        build_messages(example["summary"]), tokenize=False, add_generation_prompt=True)
    answer = answer_text(example["label"]) + tokenizer.eos_token
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def pad_batch(batch: List[Dict[str, List[int]]], pad_token_id: int) -> dict:
    """Make every example in a batch the same length."""
    import torch

    longest = max(len(item["input_ids"]) for item in batch)
    padded = {"input_ids": [], "attention_mask": [], "labels": []}
    fill = {"input_ids": pad_token_id, "attention_mask": 0, "labels": IGNORE_INDEX}
    for item in batch:
        for key, fill_value in fill.items():
            padded[key].append(item[key] + [fill_value] * (longest - len(item[key])))
    return {key: torch.tensor(values) for key, values in padded.items()}


def load_model_and_tokenizer(base_model: str):
    """Load the base model in 4-bit and attach small trainable LoRA layers."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantisation = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,       # a T4 cannot do bfloat16
        bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=quantisation, device_map={"": 0})
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def last_checkpoint(output_dir: Path) -> Optional[str]:
    """The newest saved checkpoint, so a Colab disconnect can be resumed."""
    checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(checkpoints[-1]) if checkpoints else None


def train(output_dir: Path, base_model: str, epochs: int, limit: Optional[int]) -> None:
    """Fine-tune and save the adapter into output_dir."""
    from transformers import Trainer, TrainingArguments

    model, tokenizer = load_model_and_tokenizer(base_model)
    train_examples = load_split(DATA_FOLDER / "train.jsonl")[:limit]
    validation_examples = load_split(DATA_FOLDER / "validation.jsonl")[: (limit or 300)]
    train_data = [tokenize_example(e, tokenizer) for e in train_examples]
    validation_data = [tokenize_example(e, tokenizer) for e in validation_examples]
    arguments = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=epochs, learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE, gradient_accumulation_steps=ACCUMULATION_STEPS,
        fp16=True, logging_steps=10, save_steps=SAVE_EVERY_STEPS, eval_steps=SAVE_EVERY_STEPS,
        eval_strategy="steps", save_total_limit=2, report_to="none",
        lr_scheduler_type="cosine", warmup_ratio=0.05, remove_unused_columns=False)
    trainer = Trainer(model=model, args=arguments, train_dataset=train_data,
                      eval_dataset=validation_data,
                      data_collator=lambda batch: pad_batch(batch, tokenizer.pad_token_id))
    trainer.train(resume_from_checkpoint=last_checkpoint(output_dir))
    model.save_pretrained(str(output_dir / "adapter"))
    tokenizer.save_pretrained(str(output_dir / "adapter"))
    print(f"Adapter saved to {output_dir / 'adapter'}")


def main() -> None:
    """Read the options and start training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, default=None, help="use only N examples (quick smoke test)")
    args = parser.parse_args()
    train(Path(args.output_dir), args.base_model, args.epochs, args.limit)


if __name__ == "__main__":
    main()
