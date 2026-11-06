# PRD: AML Alert Investigator

Status: planning (docs first, code next) · Owner: Uttam · Last updated: 6 Nov 2026

## 1. Problem

Banks generate large numbers of anti-money-laundering (AML) alerts. Most are false positives, and analysts spend hours reading transaction summaries to decide which few deserve a closer look. Simple threshold rules over-flag legitimate customers (a cash-heavy shop, a house-sale payout, salary followed by rent) and miss patterns that are split up to stay under a limit.

## 2. Users

| User | Need |
| --- | --- |
| AML analyst (primary) | A fast, explainable first read of each alert: what pattern it looks like, how risky, and why, with the evidence one click away and the ability to overrule |
| AML team lead | Fewer false positives reaching analysts, and an audit trail of every decision |
| Reviewer (hiring manager or technical peer) | Evidence that the model was fine-tuned and evaluated honestly against baselines |

## 3. Goals and non-goals

Goals
- Generate a reproducible synthetic dataset with planted AML typologies, benign activity and hard negatives.
- Fine-tune a small open model (QLoRA) to return a typology, a risk level and a one-line reason for each alert.
- Compare it honestly with a rules baseline and the un-tuned base model, on data the model never saw.
- Wrap it in a fixed investigation workflow with analyst approval, override and an audit log.
- Publish the model and dataset with clear cards.

Non-goals (v1)
- Real customer or transaction data. Everything is synthetic.
- Regulatory compliance. This is a prototype, not certified for any regulator's requirements; thresholds are illustrative and configurable.
- Accusations. The system says "refer for review", never "fraud" or "guilty".
- Autonomous action. Nothing is filed, frozen or sent without an analyst.
- Free-form autonomous planning by an LLM (small models call tools unreliably; a fixed workflow is the v1 design).
- Models above 3B parameters, or full fine-tuning.

## 4. Functional requirements

| ID | Requirement | Acceptance criteria |
| --- | --- | --- |
| FR1 | Synthetic data generator | Deterministic (seed 42). Plants 4 typologies, benign cases and hard negatives. Regenerates byte-identically |
| FR2 | Feature computation | Python computes every count, amount, window and ratio; the alert summary text is rendered from those features. No model produces a number |
| FR3 | Leak-free splits | Every customer belongs to exactly one set. The held-out set also uses wording templates and amount ranges reserved for it, and is never used for training or tuning |
| FR4 | Rules baseline | Configurable threshold rules, each with a rule id and reason, scored on the same test and held-out sets |
| FR5 | Fine-tuned classifier | QLoRA fine-tune of the base model. Output is JSON: `typology`, `risk`, `reason` |
| FR6 | Output validation | JSON parses; typology and risk are from the fixed lists; reason is at most 200 characters and every number in it appears in the input. An invalid output is retried once, then routed to "needs review" |
| FR7 | Evaluation report | Systems A to D (below) scored on test and held-out. Per-typology precision, recall, F1; macro-F1; false-positive rate on benign and hard negatives; JSON validity rate; faithfulness rate |
| FR8 | Investigation workflow | Six fixed steps: load alert, fetch transactions, compute velocity and profile comparison, classify, draft case note, analyst decision |
| FR9 | Analyst review | Approve or override the verdict. The original alert and the model's verdict are preserved; the override is an overlay with time and note |
| FR10 | Guardrails | Banned wording ("fraud", "criminal", "guilty") is rejected in model output and case notes; no step with side effects runs without analyst approval |
| FR11 | Export | Case note downloadable as Markdown and JSON |
| FR12 | Publication | Model (adapter and merged) and dataset on Hugging Face, each with a card stating the data is synthetic |

Labels
- Typologies: `STRUCTURING`, `RAPID_PASS_THROUGH`, `MULE_ACCOUNT`, `ROUND_TRIPPING`, `NO_CONCERN`.
- Risk: `low`, `medium`, `high`.

Systems compared
- A: threshold rules.
- B: base model, prompted only.
- C: fine-tuned model.
- D (optional): prompted Claude, if the owner provides an API key.

## 5. Non-functional requirements

- Determinism: data, features, splits and the rules baseline are reproducible.
- Honesty: report results on the held-out set even when the fine-tuned model does not win; say plainly that the data is synthetic.
- Explainability: every verdict carries a reason, and the evidence behind it is one click away.
- Privacy: synthetic data only; training happens on Colab with that data; no real customer data anywhere.
- Speed: seconds per alert on a GPU. On a CPU it will be slower, so the demo app can show precomputed verdicts for the sample alerts.
- Readable code: plain names, short functions, docstrings, named constants and sentence-style test names (rules in AGENTS.md), so the owner can read, run and explain every file.
- Repository size: the laptop needs no heavy training libraries; training dependencies live in a separate Colab requirements file.

## 6. Success metrics

These are targets, not results. Results are recorded in the README only after the evaluation has been run.

| Metric | Target |
| --- | --- |
| JSON validity rate (fine-tuned) | at least 99% |
| Macro-F1 on held-out, fine-tuned vs rules baseline | fine-tuned higher |
| False-positive rate on hard negatives, fine-tuned vs rules baseline | fine-tuned lower |
| Faithfulness (numbers in reason appear in input) | at least 98% |
| Automated tests | all pass |

If a target is missed, the README says so and explains why.

## 7. Data design

| Set | Rows | Purpose |
| --- | --- | --- |
| Train | 2,400 | About 350 per typology (1,400), 700 benign, 300 hard negatives |
| Validation | 300 | Tuning |
| Test | 400 | Score under normal conditions |
| Held-out | 300 | New wording, new amount ranges, new customers. Not touched until the final run |

Hard negatives are legitimate cases that look suspicious: a real cash-heavy business, a house-sale payout, salary followed by rent, repaying a friend.

## 8. Open decisions

1. Base model: Qwen2.5-1.5B-Instruct first; move to about 3B only if results are weak. Check the base model's licence before publishing the merged weights.
2. Prompted-Claude baseline (system D): include or skip.
3. Repository visibility: private until polished; make the Hugging Face model public when applying for roles.

## 9. Roadmap

| Week | Work |
| --- | --- |
| 0 | Practice fine-tune on a toy dataset in Colab |
| 1 | Generator, features, rules baseline, dataset |
| 2 | First fine-tune; fix JSON failures |
| 3 | Full evaluation, held-out run, error analysis |
| 4 | Investigation workflow and Streamlit app |
| 5 | Publish to Hugging Face, README, tests, write-up |

After v1: an LLM planner for the investigation workflow, more typologies, and a larger base model.

## 10. Risks

| Risk | Mitigation |
| --- | --- |
| Model memorises templates | Several wording templates; templates reserved for the held-out set |
| Invalid JSON | Schema check, one retry, report the validity rate |
| Over-flagging | Hard negatives; track the false-positive rate |
| Colab disconnects | Checkpoint to Google Drive every few steps |
| Overclaiming | State that the data is synthetic; no real-world accuracy claims |
| Base-model licence | Check before publishing merged weights; publish the adapter if unsure |
