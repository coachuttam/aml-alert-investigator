"""The fixed lists of labels that every part of the project agrees on.

Output validation (JSON checks, banned wording, faithfulness) is added here
later, when the model output step is built.
"""

# The four money-laundering patterns the model must recognise.
TYPOLOGIES = (
    "STRUCTURING",         # cash split into deposits just under a reporting limit
    "RAPID_PASS_THROUGH",  # large money in, almost all of it out again within days
    "MULE_ACCOUNT",        # many unrelated senders, then a quick cash-out
    "ROUND_TRIPPING",      # money sent out comes back through other parties
)

# The answer for normal activity, including legitimate look-alikes.
NO_CONCERN = "NO_CONCERN"

ALL_LABELS = TYPOLOGIES + (NO_CONCERN,)

RISK_LEVELS = ("low", "medium", "high")

# Every customer belongs to exactly one of these sets.
SPLITS = ("train", "validation", "test", "heldout")
