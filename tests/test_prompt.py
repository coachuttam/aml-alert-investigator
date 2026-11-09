"""Checks the prompt and the way training text is turned into token ids (no GPU needed)."""
import json

from src.finetune import IGNORE_INDEX, tokenize_example
from src.prompt import SYSTEM_PROMPT, answer_text, build_messages
from src.schema import ALL_LABELS


class FakeTokenizer:
    """One 'token' per character, so the tests do not need to download a model."""
    eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "".join(m["content"] for m in messages) + "<assistant>"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": [ord(c) for c in text]}


EXAMPLE = {"summary": "Summary text.",
           "label": {"typology": "STRUCTURING", "risk": "high", "reason": "9 deposits."}}


def test_the_system_prompt_names_every_label():
    for label in ALL_LABELS:
        assert label in SYSTEM_PROMPT


def test_the_messages_are_system_then_the_alert_summary():
    messages = build_messages("hello")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "hello"


def test_the_trained_answer_is_valid_json_with_the_three_fields():
    answer = json.loads(answer_text(EXAMPLE["label"]))
    assert list(answer) == ["typology", "risk", "reason"]


def test_loss_is_switched_off_for_the_prompt_and_on_for_the_answer():
    tokens = tokenize_example(EXAMPLE, FakeTokenizer())
    answer_length = len(answer_text(EXAMPLE["label"]) + "<eos>")
    assert all(x == IGNORE_INDEX for x in tokens["labels"][:-answer_length])
    assert all(x != IGNORE_INDEX for x in tokens["labels"][-answer_length:])
    assert len(tokens["labels"]) == len(tokens["input_ids"])


def test_long_examples_are_cut_to_the_maximum_length():
    tokens = tokenize_example(EXAMPLE, FakeTokenizer(), max_length=20)
    assert len(tokens["input_ids"]) == 20


def test_warmup_is_at_least_one_step_and_about_five_percent_of_training():
    from src.finetune import count_warmup_steps
    assert count_warmup_steps(64, 1) == 1
    assert count_warmup_steps(2400, 2) == 15      # 150 steps per epoch x 2 epochs x 5%
