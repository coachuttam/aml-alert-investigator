# AGENTS.md

Instructions for AI coding agents (Claude Code, Cursor, Codex) working in this repo. Read ARCHITECTURE.md for how it fits together and PRD.md for what it must do.

## Setup and checks

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # light: data, rules, evaluation, app
python -m pytest -q                      # must stay green
python -m src.generator                  # regenerate data/raw/ (deterministic, seed 42; only validation, test and held-out are committed, train.jsonl is not)
python -m src.dataset                    # rebuild data/*.jsonl (summaries, labels, features; committed)
python -m src.evaluate                   # report test and held-out results after any change
streamlit run app.py
```

Training runs on Colab, not the laptop. Heavy libraries (`torch`, `transformers`, `peft`, `trl`, `bitsandbytes`) go in `requirements-train.txt`, never in `requirements.txt`.

## Hard rules

1. **Never tune on the held-out set** (`data/heldout.jsonl`). Do not read its summaries to write rules, templates or prompts. Run it once at the end. The two held-out wording templates (`report` and `bullets` in `templates.py`) are fixed: do not edit them after seeing any result. Tune risk cut-offs, thresholds and rules on the training set only. If held-out results change, say why. To test new ideas honestly, add new held-out templates and customers to the generator before changing anything else.
2. **The model labels and explains; Python counts.** No model may output an amount, date or count that is not already in its input. `schema.py` enforces this with the faithfulness check.
3. **Split before generating.** Customers are assigned to a split first, and each customer belongs to exactly one. Templates reserved for the held-out set must never appear in training, validation or test.
4. **Say "refer for review", never "fraud".** Banned wording (fraud, criminal, guilty) is rejected in model output and in user-facing text.
5. **Keep the original.** Never overwrite an alert or the model's verdict; analyst decisions are overlays with time and note.
6. **No side effects without approval.** Nothing is filed, frozen or sent before the analyst's decision.
7. **Synthetic data only.** Never add real transactions, names or account numbers.
8. **Report honestly.** If the fine-tuned model does not beat the baselines, the README says so.

## Readable code

The owner will read, run and explain every file, so write for a reader, not for cleverness.

- Plain, descriptive names (`count_cash_deposits_in_window`, not `ccdw`). No abbreviations a newcomer would have to guess.
- One job per function, about 20 lines or fewer. If a function needs a long comment to explain its steps, split it.
- A short docstring on every module and public function: what it does, in plain English, with a tiny example where it helps.
- Comments explain **why**, not what. Never restate the code.
- No magic numbers. Thresholds, window lengths and seeds are named constants at the top of the file, each with a one-line reason.
- Prefer simple loops and clear `if` statements over dense one-liners and nested comprehensions.
- Type hints and dataclasses for anything passed between modules.
- Consistent formatting with `ruff format` (or `black`), and `ruff check` clean.
- Test names read as sentences: `test_split_never_puts_a_customer_in_two_sets`.
- Keep each file focused and under about 300 lines; split a file before it sprawls.

## Conventions

- Python 3.9+ compatible, because the owner's Mac environment may be 3.9 while Colab is newer: start files with `from __future__ import annotations`, use `typing.List` and `Optional`, and avoid `match` statements and `X | Y` unions at runtime. Type hints and dataclasses.
- Keep functions small; no new dependency without a reason.
- Change the generator, then rerun `python -m src.generator`, then tests and evaluation. Output is deterministic.
- New typology: generator pattern, feature, label list in `schema.py`, rule, test, then retrain.
- New rule: needs a rule id and a plain-English reason.
- UI text and layout follow DESIGN_SYSTEM.md.
- Keep prompts to the model short and identical between training and inference.

## Definition of done

Tests pass, evaluation run and both test and held-out numbers reported, README updated if behaviour changed, no held-out tuning.
