"""Checks the folder built for the Hugging Face Space (no network)."""
from __future__ import annotations

import re

from src import build_space


def built(tmp_path):
    return build_space.build_space_folder(tmp_path / "space")


def test_the_built_folder_has_the_app_the_data_and_the_saved_answers(tmp_path):
    folder = built(tmp_path)
    for name in ("app.py", "README.md", "requirements.txt", "src/review.py", "src/investigator.py",
                 "data/train.jsonl", "data/raw/test.jsonl", "results/finetuned_heldout.jsonl"):
        assert (folder / name).exists(), name


def test_training_and_publishing_scripts_and_the_raw_training_set_stay_out(tmp_path):
    folder = built(tmp_path)
    for name in ("src/finetune.py", "src/predict.py", "src/publish.py", "data/raw/train.jsonl", "audit"):
        assert not (folder / name).exists(), name


def test_the_space_readme_declares_a_docker_space_and_states_the_data_is_synthetic(tmp_path):
    text = (built(tmp_path) / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "sdk: docker" in text and "app_port: 8501" in text
    assert "Synthetic data only" in text
    assert "did not beat them on held-out" in text


def test_the_dockerfile_installs_the_requirements_and_starts_streamlit_on_the_declared_port(tmp_path):
    folder = built(tmp_path)
    dockerfile = (folder / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements.txt" in dockerfile and "streamlit" in dockerfile and "--server.port=8501" in dockerfile
    assert "streamlit==" in (folder / "requirements.txt").read_text(encoding="utf-8")


def test_no_token_or_key_is_in_the_built_folder(tmp_path):
    secret_pattern = re.compile(r"(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_|sk-[A-Za-z0-9]{20,})")
    for path in built(tmp_path).rglob("*"):
        if path.is_file():
            assert not secret_pattern.search(path.read_text(encoding="utf-8", errors="ignore")), path
