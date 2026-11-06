# Design system: AML Alert Investigator UI

The interface serves an AML analyst who must trust and defend every decision. Principles: evidence one click away, uncertainty shown before verdicts, calm language. The look follows the Statement Analyser (P10) so the two projects feel like one family.

## Layout

| Area | Content |
| --- | --- |
| Sidebar | Source (sample alerts or upload), load button, risk filter, typology filter. Synthetic-data note. |
| Header | Alert id · customer · profile (occupation, declared income) · window · transaction count |
| Notice banner | Amber when the model output was invalid and the alert is in "needs review". None when clean. |
| Tabs | Alerts (n) · Alert detail · Evaluation · Audit log (n). Counts in tab labels. |

## Components

**Alert card (queue)**: alert id, customer, one-line summary, verdict chip, risk badge. Sorted by risk, then by date.

**Verdict chip**: typology as plain words ("Structuring", "Rapid pass-through", "Mule account", "Round-tripping", "No concern"), with the source in small text ("Fine-tuned model", "Rules", "Analyst").

**Risk badge**: pill, 0.7rem, weight 600.

| Risk | Colour | Text |
| --- | --- | --- |
| low | green tint | "low" |
| medium | amber tint | "medium" |
| high | red tint | "high" |

**Evidence table**: columns Date, Channel, Amount (signed, + credit / − debit), Counterparty, Note. Rows that support the verdict are highlighted. A caption lists the computed features (counts, totals, ratio) with the same numbers the model saw.

**Reason line**: the model's one-line reason under the verdict, with a small "numbers checked against the input" tick when the faithfulness check passed.

**Decision panel**: Approve, or Override with a typology dropdown, a risk dropdown and a required note. Shows "Model said X" next to "Analyst set Y". The original alert and the model's verdict stay visible.

**KPI card (Evaluation tab)**: label (0.78rem, 70% opacity), value (1.45rem, weight 650), one-line sub-text. 1px border at 25% grey, 10px radius, min-height 132px. Cards: macro-F1, false-positive rate on hard negatives, JSON validity, faithfulness.

**Results table (Evaluation tab)**: systems as rows (A rules, B base model, C fine-tuned, D prompted Claude if run), metrics as columns, test and held-out side by side. The held-out column is labelled "held-out (run once)".

**Audit log**: one line per event with time, alert id, action, before → after and note.

## Numbers and dates

- Currency: ₹ with Indian digit grouping (₹1,02,000; ₹3,00,000). Negatives as -₹9,900 (sign before the symbol).
- Dates: "05 May 2026" in tables; "01 May to 06 May 2026" for windows.
- Percentages: one decimal (94.0%).

## Charts

- Confusion matrix for each system on the held-out set, same colour scale, counts in the cells.
- Bar chart of per-typology F1, one bar per system, no more than three systems per chart.
- Streamlit default palette.

## Language

- Say what the analyst should do: "Review the 11 cash deposits before approving."
- "Refer for review", never "fraud", "criminal" or "guilty". The system describes a pattern, not a person.
- Explain a verdict in plain words: "11 cash deposits just under the limit, then 94% moved out within 48 hours."
- Mark uncertainty honestly: "Needs review: the model's answer could not be validated."
- Every screen with model output carries a small note: "Synthetic data. Prototype, not certified for compliance."

## Accessibility

- Never rely on colour alone: risk badges carry text; highlighted evidence rows also carry a marker column.
- Keep contrast at Streamlit defaults; test both light and dark themes before release.
