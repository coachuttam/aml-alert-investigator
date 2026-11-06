"""Made-up names for counterparties.

Two separate pools exist so the held-out set contains names the model never
saw in training. All names are invented and generic.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class NamePool:
    """A set of first names, last names and business endings."""

    first_names: Tuple[str, ...]
    last_names: Tuple[str, ...]
    business_endings: Tuple[str, ...]

    def all_people(self) -> List[str]:
        """Every first-name and last-name combination, in a fixed order."""
        return [f"{first} {last}" for first in self.first_names for last in self.last_names]

    def person(self, rng: random.Random) -> str:
        """One person's name, for example 'Ravi Kumar'."""
        return rng.choice(self.all_people())

    def people(self, rng: random.Random, count: int) -> List[str]:
        """`count` different people (no name appears twice)."""
        return rng.sample(self.all_people(), count)

    def business(self, rng: random.Random) -> str:
        """One business name, for example 'Patel Traders'."""
        return f"{rng.choice(self.last_names)} {rng.choice(self.business_endings)}"


STANDARD_NAMES = NamePool(
    first_names=("Ravi", "Sunita", "Manoj", "Kavita", "Arjun", "Deepa", "Suresh", "Meena",
                 "Rohit", "Anjali", "Vijay", "Pooja", "Kiran", "Neha", "Sanjay", "Lata"),
    last_names=("Kumar", "Sharma", "Patel", "Reddy", "Nair", "Iyer", "Gupta", "Singh", "Das", "Joshi"),
    business_endings=("Traders", "Enterprises", "Stores", "Agencies"),
)

# Used only for the held-out set: no first name, last name or ending is shared with STANDARD_NAMES.
HELDOUT_NAMES = NamePool(
    first_names=("Farhan", "Gayatri", "Harish", "Ishita", "Jayant", "Komal", "Lakshman", "Mansi",
                 "Naveen", "Pranav", "Radha", "Sameer", "Tanvi", "Umesh", "Vidya", "Yash"),
    last_names=("Banerjee", "Chawla", "Desai", "Fernandes", "Gowda", "Hegde", "Kapoor", "Menon",
                "Pillai", "Rastogi"),
    business_endings=("Logistics", "Ventures", "Distributors", "Services"),
)
