"""Generates the synthetic alerts: customers, 30-day windows and true labels.

Run it to write the raw alerts to data/raw/:
    python -m src.generator

The output is deterministic: the same seed always gives the same alerts. Each
alert gets its own random generator, so changing one alert never shifts the others.
Every customer appears in exactly one alert, so no customer can sit in two splits.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.models import Customer, RawAlert, Transaction
from src.names import HELDOUT_NAMES, STANDARD_NAMES, NamePool
from src.patterns import (
    HARD_NEGATIVE_PLANTERS,
    TYPOLOGY_PLANTERS,
    Ledger,
    add_everyday_activity,
    add_one_off,
    round_to,
)
from src.schema import NO_CONCERN, SPLITS, TYPOLOGIES

SEED = 42
WINDOW_DAYS = 30

# Windows end between 1 Mar and 31 Oct 2026, so alerts do not all share one calendar.
WINDOW_END_EARLIEST = date(2026, 3, 1)
WINDOW_END_SPREAD_DAYS = 244

# Declared monthly income range in rupees for each occupation.
OCCUPATION_INCOME: Dict[str, Tuple[int, int]] = {
    "salaried employee": (30_000, 90_000),
    "shopkeeper": (40_000, 120_000),
    "small trader": (50_000, 150_000),
    "freelancer": (25_000, 80_000),
    "delivery driver": (18_000, 35_000),
    "homemaker": (10_000, 25_000),
    "student": (8_000, 15_000),
}
ALL_OCCUPATIONS = list(OCCUPATION_INCOME)
BUSINESS_OCCUPATIONS = ["shopkeeper", "small trader", "freelancer"]

# Which occupations each pattern may use. Mules skew to low incomes, as in real cases.
TYPOLOGY_OCCUPATIONS: Dict[str, List[str]] = {
    "STRUCTURING": ALL_OCCUPATIONS,
    "RAPID_PASS_THROUGH": ALL_OCCUPATIONS,
    "MULE_ACCOUNT": ["student", "homemaker", "delivery driver", "salaried employee"],
    "ROUND_TRIPPING": BUSINESS_OCCUPATIONS,
    NO_CONCERN: ALL_OCCUPATIONS,
}
HARD_NEGATIVE_OCCUPATIONS: Dict[str, List[str]] = {
    "cash_heavy_shop": ["shopkeeper", "small trader"],
    "house_sale": ALL_OCCUPATIONS,
    "salary_then_commitments": ["salaried employee"],
    "repay_friend": ALL_OCCUPATIONS,
}
HARD_NEGATIVE_KINDS = list(HARD_NEGATIVE_PLANTERS)

# A mule account is usually new. Everyone else has had their account for a year or more.
MULE_ACCOUNT_AGE_MONTHS = (1, 6)
NORMAL_ACCOUNT_AGE_MONTHS = (12, 120)


@dataclass(frozen=True)
class SplitPlan:
    """How many alerts of each kind one split holds, and how it differs from the others."""

    name: str
    per_typology: int          # alerts for each of the four patterns
    benign: int                # ordinary activity
    hard_negatives: int        # legitimate look-alikes, shared equally between the four kinds
    income_scale: float        # 1.0 for normal amounts, larger for the held-out set
    template_group: str        # which wording templates may describe these alerts
    name_pool: NamePool


SPLIT_PLANS: Dict[str, SplitPlan] = {
    "train": SplitPlan("train", 350, 700, 300, 1.0, "shared", STANDARD_NAMES),
    "validation": SplitPlan("validation", 45, 80, 40, 1.0, "shared", STANDARD_NAMES),
    "test": SplitPlan("test", 60, 100, 60, 1.0, "shared", STANDARD_NAMES),
    # The held-out set uses new names, higher amounts and reserved wording templates.
    "heldout": SplitPlan("heldout", 45, 60, 60, 1.6, "heldout", HELDOUT_NAMES),
}


def make_customer(rng: random.Random, customer_id: str, typology: str,
                  hard_negative_kind: Optional[str], income_scale: float) -> Customer:
    """Pick an occupation that fits the pattern, then an income and account age."""
    if hard_negative_kind:
        occupations = HARD_NEGATIVE_OCCUPATIONS[hard_negative_kind]
    else:
        occupations = TYPOLOGY_OCCUPATIONS[typology]
    occupation = rng.choice(occupations)
    low, high = OCCUPATION_INCOME[occupation]
    income = round_to(rng.uniform(low, high) * income_scale, 1_000)
    age_range = MULE_ACCOUNT_AGE_MONTHS if typology == "MULE_ACCOUNT" else NORMAL_ACCOUNT_AGE_MONTHS
    return Customer(customer_id, occupation, income, rng.randint(*age_range))


def generate_alert(plan: SplitPlan, index: int, typology: str,
                   hard_negative_kind: Optional[str]) -> RawAlert:
    """Build one alert: everyday activity first, then the planted pattern."""
    rng = random.Random(f"{SEED}:{plan.name}:{index}")
    number = f"{plan.name}-{index + 1:04d}"
    customer = make_customer(rng, f"C-{number}", typology, hard_negative_kind, plan.income_scale)
    window_end = WINDOW_END_EARLIEST + timedelta(days=rng.randint(0, WINDOW_END_SPREAD_DAYS))
    window_start = window_end - timedelta(days=WINDOW_DAYS - 1)

    ledger = Ledger(f"A-{number}")
    pool = plan.name_pool
    needs_rent = hard_negative_kind != "salary_then_commitments"
    add_everyday_activity(ledger, rng, customer, window_start, window_end, pool, include_rent=needs_rent)

    if hard_negative_kind:
        planter = HARD_NEGATIVE_PLANTERS[hard_negative_kind]
        planter(ledger, rng, customer, window_start, window_end, pool, plan.income_scale)
    elif typology == NO_CONCERN:
        add_one_off(ledger, rng, customer, window_start, window_end, pool)
    else:
        planter = TYPOLOGY_PLANTERS[typology]
        planter(ledger, rng, customer, window_start, window_end, pool, plan.income_scale)

    return RawAlert(
        alert_id=f"A-{number}", split=plan.name, template_group=plan.template_group,
        customer=customer, window_start=window_start, window_end=window_end,
        transactions=ledger.finish(), typology=typology, hard_negative_kind=hard_negative_kind,
    )


def build_recipes(plan: SplitPlan) -> List[Tuple[str, Optional[str]]]:
    """The list of (typology, hard-negative kind) for a split, in shuffled order."""
    recipes: List[Tuple[str, Optional[str]]] = []
    for typology in TYPOLOGIES:
        recipes += [(typology, None)] * plan.per_typology
    recipes += [(NO_CONCERN, None)] * plan.benign
    for number in range(plan.hard_negatives):
        recipes.append((NO_CONCERN, HARD_NEGATIVE_KINDS[number % len(HARD_NEGATIVE_KINDS)]))
    random.Random(f"{SEED}:{plan.name}:order").shuffle(recipes)
    return recipes


def generate_split(plan: SplitPlan) -> List[RawAlert]:
    """All alerts for one split."""
    recipes = build_recipes(plan)
    return [generate_alert(plan, index, typology, kind) for index, (typology, kind) in enumerate(recipes)]


def generate_all() -> Dict[str, List[RawAlert]]:
    """All four splits, keyed by split name."""
    return {name: generate_split(SPLIT_PLANS[name]) for name in SPLITS}


# --------------------------------------------------------------------------
# Saving and loading (one JSON line per alert)
# --------------------------------------------------------------------------

def alert_to_dict(alert: RawAlert) -> dict:
    """A plain dictionary that json can write."""
    customer = alert.customer
    return {
        "alert_id": alert.alert_id,
        "split": alert.split,
        "template_group": alert.template_group,
        "typology": alert.typology,
        "hard_negative_kind": alert.hard_negative_kind,
        "window_start": alert.window_start.isoformat(),
        "window_end": alert.window_end.isoformat(),
        "customer": {
            "customer_id": customer.customer_id,
            "occupation": customer.occupation,
            "declared_monthly_income": customer.declared_monthly_income,
            "account_age_months": customer.account_age_months,
        },
        "transactions": [
            {"txn_id": t.txn_id, "day": t.day.isoformat(), "amount": t.amount,
             "direction": t.direction, "channel": t.channel, "counterparty": t.counterparty}
            for t in alert.transactions
        ],
    }


def alert_from_dict(data: dict) -> RawAlert:
    """The reverse of alert_to_dict."""
    transactions = [
        Transaction(t["txn_id"], date.fromisoformat(t["day"]), t["amount"],
                    t["direction"], t["channel"], t["counterparty"])
        for t in data["transactions"]
    ]
    return RawAlert(
        alert_id=data["alert_id"], split=data["split"], template_group=data["template_group"],
        customer=Customer(**data["customer"]),
        window_start=date.fromisoformat(data["window_start"]),
        window_end=date.fromisoformat(data["window_end"]),
        transactions=transactions, typology=data["typology"],
        hard_negative_kind=data["hard_negative_kind"],
    )


def save_raw_alerts(alerts_by_split: Dict[str, List[RawAlert]], folder: Path) -> None:
    """Write one .jsonl file per split into `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    for split, alerts in alerts_by_split.items():
        lines = [json.dumps(alert_to_dict(alert), sort_keys=True) for alert in alerts]
        (folder / f"{split}.jsonl").write_text("\n".join(lines) + "\n")


def load_raw_alerts(path: Path) -> List[RawAlert]:
    """Read one .jsonl file written by save_raw_alerts."""
    return [alert_from_dict(json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    """Generate every split and write it to data/raw/."""
    folder = Path(__file__).resolve().parent.parent / "data" / "raw"
    alerts_by_split = generate_all()
    save_raw_alerts(alerts_by_split, folder)
    for split, alerts in alerts_by_split.items():
        print(f"{split:<11} {len(alerts):>5} alerts")
    print(f"Written to {folder}")


if __name__ == "__main__":
    main()
