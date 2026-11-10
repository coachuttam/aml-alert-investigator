"""Builds the folder for the free Hugging Face Space demo, and uploads it.

    python -m src.build_space              builds space_build/ so you can look at it first
    python -m src.build_space --upload     builds it, then uploads it as a public Space
    python -m src.build_space --upload --private

Hugging Face now needs a PRO subscription for Docker Spaces (402 error on the free plan).
Streamlit Community Cloud is the free alternative; see README.md.

The Space runs the review app on saved model answers, so it needs no GPU. The token
is read from the HF_TOKEN environment variable, never typed into a file.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.publish import MODEL_REPO, DATASET_REPO, USERNAME, get_api

SPACE_REPO = f"{USERNAME}/aml-alert-investigator-demo"
ROOT = Path(__file__).resolve().parent.parent
BUILD_FOLDER = ROOT / "space_build"

# The Space only reads saved answers, so the training and publishing scripts stay out.
LEFT_OUT_FROM_SRC = {"finetune.py", "predict.py", "publish.py", "build_space.py"}
RAW_SPLITS_SHOWN = ("validation", "test", "heldout")   # the app never reads the raw training set
STREAMLIT_VERSION = "1.50.0"                           # the version tested on the laptop
SPACE_REQUIREMENTS = f"streamlit=={STREAMLIT_VERSION}\npandas>=2.0\n"
APP_PORT = 8501                                        # Streamlit's default port

# Hugging Face no longer has a built-in Streamlit option, so the Space runs in a small Docker image.
# It runs as user 1000, as Hugging Face expects, so the app can write its audit folder.
DOCKERFILE = f"""FROM python:3.10-slim
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
EXPOSE {APP_PORT}
CMD ["streamlit", "run", "app.py", "--server.port={APP_PORT}", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
"""

SPACE_README = f"""---
title: AML Alert Investigator
sdk: docker
app_port: {APP_PORT}
license: mit
short_description: Review AML alerts with a fine-tuned model (synthetic data)
---

# AML Alert Investigator (demo)

A review screen for anti-money-laundering alerts. A Qwen2.5-1.5B model fine-tuned with QLoRA
labels each alert (typology, risk, one-line reason); Python computes every number; a fixed
six-step workflow drafts a case note; an analyst approves or overrides. The demo shows
**saved** model answers, so no GPU runs here.

**Synthetic data only. Prototype, not certified for compliance.**

Honest results (macro-F1, test / held-out): tuned rules 0.990 / 0.990, fine-tuned model
1.000 / 0.936, base model 0.135 / 0.181. The fine-tuned model matched the tuned rules on
test but did not beat them on held-out wording.

This is a shared demo: saved decisions are visible to other visitors and reset when the
Space restarts. Please do not type personal information.

Links: [model](https://huggingface.co/{MODEL_REPO}) ·
[dataset](https://huggingface.co/datasets/{DATASET_REPO}) ·
[code](https://github.com/{USERNAME}/aml-alert-investigator)
"""


def copy_file(source: Path, target_folder: Path) -> None:
    """Copy one file into the build folder, keeping its place in the project."""
    destination = target_folder / source.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def files_for_space() -> list:
    """Every project file the demo needs."""
    files = [ROOT / "app.py"]
    files += [path for path in sorted((ROOT / "src").glob("*.py")) if path.name not in LEFT_OUT_FROM_SRC]
    files += sorted((ROOT / "data").glob("*.jsonl"))
    files += [ROOT / "data" / "raw" / f"{split}.jsonl" for split in RAW_SPLITS_SHOWN]
    files += sorted((ROOT / "results").glob("*.jsonl"))
    return files


def build_space_folder(target_folder: Path = BUILD_FOLDER) -> Path:
    """Make a clean build folder, ready to upload."""
    if target_folder.exists():
        shutil.rmtree(target_folder)
    target_folder.mkdir(parents=True)
    for source in files_for_space():
        copy_file(source, target_folder)
    (target_folder / "README.md").write_text(SPACE_README, encoding="utf-8")
    (target_folder / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
    (target_folder / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    return target_folder


def upload_space(folder: Path, private: bool) -> None:
    """Create the Space if needed and upload the folder to it."""
    api = get_api()
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=SPACE_REPO, repo_type="space")
    print(f"Space uploaded: https://huggingface.co/spaces/{SPACE_REPO}")


def main() -> None:
    """Build the folder, and upload it when asked."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="upload to Hugging Face after building")
    parser.add_argument("--private", action="store_true", help="make the Space private (default: public)")
    args = parser.parse_args()
    folder = build_space_folder()
    print(f"Built {folder} with {sum(1 for path in folder.rglob('*') if path.is_file())} files.")
    if args.upload:
        upload_space(folder, private=args.private)


if __name__ == "__main__":
    main()
