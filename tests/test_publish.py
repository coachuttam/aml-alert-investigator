"""Checks the Hugging Face cards and the safety of the publish script (no network)."""
import pytest

from src import publish


def test_both_cards_exist_and_start_with_a_metadata_block():
    for name in ("MODEL_CARD.md", "DATASET_CARD.md"):
        text = (publish.CARD_FOLDER / name).read_text(encoding="utf-8")
        assert text.startswith("---\n") and "license:" in text.split("---")[1]


def test_the_model_card_states_the_limitations_and_honest_results():
    text = (publish.CARD_FOLDER / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "did not beat well-tuned threshold rules" in text
    assert "0.936" in text and "0.990" in text


def test_publishing_needs_a_token_and_never_hard_codes_one(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        publish.get_api()
    assert "hf_" not in (publish.ROOT / "src" / "publish.py").read_text()


def test_every_dataset_split_file_exists():
    for split in publish.DATASET_SPLITS:
        assert (publish.DATA_FOLDER / f"{split}.jsonl").exists()
