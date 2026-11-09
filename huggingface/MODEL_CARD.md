---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
library_name: peft
tags:
  - lora
  - qlora
  - aml
  - anti-money-laundering
  - financial-crime
  - classification
datasets:
  - coachuttam/aml-alert-investigator-data
language:
  - en
---

# AML Alert Investigator: Qwen2.5-1.5B QLoRA adapter

A small LoRA adapter for **Qwen2.5-1.5B-Instruct**. It reads a short anti-money-laundering (AML) alert summary and answers with JSON only:

```json
{"typology": "STRUCTURING", "risk": "high", "reason": "9 cash deposits just under the limit within 5 days ..."}
```

- `typology` is one of STRUCTURING, RAPID_PASS_THROUGH, MULE_ACCOUNT, ROUND_TRIPPING or NO_CONCERN.
- `risk` is low, medium or high.
- `reason` is one short sentence that may only use numbers that appear in the alert summary.

**Design rule: the model labels and explains, Python counts.** Every number is computed by code before the model sees it. A checker rejects any answer whose reason uses a number that is not in the summary.

**This is a portfolio prototype trained only on synthetic data. It is not for use on real customers, and it is not certified for any regulator's requirements.** It says "refer for review"; it never accuses anyone.

## How to use

The adapter needs the base model. Prompts must use the same system prompt as training (`src/prompt.py` in the code repository).

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "coachuttam/aml-alert-investigator-qwen2.5-1.5b-qlora")
```

Build the messages with the project's `build_messages(summary)`, call `apply_chat_template(..., add_generation_prompt=True)`, then generate greedily (`do_sample=False`, about 120 new tokens).

## Training

| Item | Value |
| --- | --- |
| Base model | Qwen/Qwen2.5-1.5B-Instruct (Apache 2.0 per its model page) |
| Method | QLoRA: 4-bit NF4 base, LoRA rank 16, alpha 32, dropout 0.05, all linear layers |
| Trainable parameters | 18.5 million (about 1.2% of the model) |
| Data | 2,400 synthetic training alerts (see the dataset card) |
| Loss | On the answer only; the prompt is masked |
| Settings | 2 epochs, learning rate 2e-4, effective batch 16, max length 512, fp16 |
| Hardware | One free Colab T4, about 36 minutes |
| Validation loss at the end | 0.0006 |

## Results

Test shares its wording templates with training. Held-out uses reserved wording templates and incomes 1.6 times larger, and was scored once after all choices were fixed.

| System | Macro-F1 test | Macro-F1 held-out | False alarms on normal alerts (test / held-out) | Valid JSON (test / held-out) |
| --- | --- | --- | --- | --- |
| Rules, default thresholds | 0.962 | 0.948 | 8.1% / 14.2% | n/a |
| Rules, tuned on train | 0.990 | 0.990 | 0.0% / 1.7% | n/a |
| Base model, prompted only | 0.135 | 0.181 | 100% / 100% | 88.5% / 72.0% |
| **This adapter** | **1.000** | **0.936** | **0.0% / 8.3%** | **100% / 99.7%** |

Risk-level accuracy for the adapter was 99.8% on test and 99.7% on held-out.

## Limitations (please read)

- **It did not beat well-tuned threshold rules on held-out** (0.936 against 0.990). The rules read exact computed numbers, so new wording does not affect them. The model reads text and lost accuracy on mule accounts (F1 0.83) and on the "salary then commitments" look-alike (53% wrongly flagged on held-out, 0% on test).
- **The task is easy by design.** The summary prints the numbers that decide the label, so a near-perfect test score is expected. Held-out is the more honest number.
- **The base-model baseline is weak on purpose.** It was told only the label names, with no definitions or examples. It shows what training added, not what the best prompt could do.
- **Synthetic data only.** Scores say nothing about real-world accuracy. Amounts, limits and thresholds are illustrative, not regulatory.
- The adapter was trained for one narrow output format. Other prompts are not supported.

## Links

Code, evaluation scripts and the investigator workflow: the `aml-alert-investigator` repository on GitHub (user `coachuttam`). Dataset: `coachuttam/aml-alert-investigator-data`.
