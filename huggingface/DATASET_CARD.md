---
license: apache-2.0
language:
  - en
task_categories:
  - text-classification
  - text-generation
tags:
  - aml
  - anti-money-laundering
  - synthetic
  - financial-crime
size_categories:
  - 1K<n<10K
---

# AML Alert Investigator: synthetic alert dataset

3,400 **fully synthetic** AML alert summaries with labels. Nothing in it is real: names, amounts and accounts are all generated. It was built to fine-tune and honestly evaluate a small language model that returns a typology, a risk level and a one-line reason.

## Splits

| Split | Alerts | Purpose |
| --- | --- | --- |
| train | 2,400 | Fine-tuning (350 per pattern, 700 ordinary, 300 look-alikes) |
| validation | 300 | Watching training loss |
| test | 400 | Scoring; shares wording templates with train |
| heldout | 300 | Scored once at the end; reserved wording templates, a separate name pool and incomes 1.6 times larger |

Each customer appears in exactly one alert and one split. Generation is deterministic (seed 42).

## Fields

- `alert_id`, `customer_id`, `split`
- `summary`: the alert text the model reads (one of five wording templates, named in `template_id`)
- `label`: `typology`, `risk` (low, medium, high) and `reason` (one sentence using only numbers from the summary)
- `hard_negative_kind`: for normal look-alikes (`cash_heavy_shop`, `house_sale`, `salary_then_commitments`, `repay_friend`), otherwise null
- `features`: every number behind the summary, computed in code

## Labels

STRUCTURING (cash deposits just under a reporting limit), RAPID_PASS_THROUGH (large credits sent on within days), MULE_ACCOUNT (many senders then a quick cash-out), ROUND_TRIPPING (money sent out returns near-whole) and NO_CONCERN, which includes hard negatives: legitimate activity that looks suspicious.

Risk is the amount at issue divided by declared monthly income: below 5 is low, 5 to under 15 is medium, 15 or more is high. Normal activity is always low.

## Limitations

- Synthetic data cannot show real-world accuracy. Patterns are cleaner and far more regular than real accounts.
- The reporting limit (Rs 50,000) and all thresholds are illustrative, not regulatory.
- The summaries state the deciding numbers directly, so the classification task is easier than real alert review.
- The held-out wording templates and name pool exist to test generalisation; do not use them for training if you want that test to mean anything.
