"""Publishes the trained adapter and the dataset to Hugging Face.

    python -m src.publish --adapter /path/to/adapter            private repos (default)
    python -m src.publish --adapter /path/to/adapter --public   make them public

The token is read from the HF_TOKEN environment variable, never typed into a file.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

USERNAME = "coachuttam"
MODEL_REPO = f"{USERNAME}/aml-alert-investigator-qwen2.5-1.5b-qlora"
DATASET_REPO = f"{USERNAME}/aml-alert-investigator-data"
ROOT = Path(__file__).resolve().parent.parent
CARD_FOLDER = ROOT / "huggingface"
DATA_FOLDER = ROOT / "data"
DATASET_SPLITS = ("train", "validation", "test", "heldout")


def get_api():
    """A Hugging Face client using the HF_TOKEN environment variable."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. In Colab: os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')")
    from huggingface_hub import HfApi
    return HfApi(token=token)


def publish_model(api, adapter_folder: Path, private: bool) -> None:
    """Upload the adapter files, with our model card in place of the auto-generated one."""
    api.create_repo(MODEL_REPO, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(adapter_folder), repo_id=MODEL_REPO, repo_type="model",
                      ignore_patterns=["README.md"])
    api.upload_file(path_or_fileobj=str(CARD_FOLDER / "MODEL_CARD.md"), path_in_repo="README.md",
                    repo_id=MODEL_REPO, repo_type="model")
    print(f"Model uploaded: https://huggingface.co/{MODEL_REPO}")


def publish_dataset(api, private: bool) -> None:
    """Upload the four split files and the dataset card."""
    api.create_repo(DATASET_REPO, repo_type="dataset", private=private, exist_ok=True)
    for split in DATASET_SPLITS:
        api.upload_file(path_or_fileobj=str(DATA_FOLDER / f"{split}.jsonl"),
                        path_in_repo=f"data/{split}.jsonl", repo_id=DATASET_REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(CARD_FOLDER / "DATASET_CARD.md"), path_in_repo="README.md",
                    repo_id=DATASET_REPO, repo_type="dataset")
    print(f"Dataset uploaded: https://huggingface.co/datasets/{DATASET_REPO}")


def main() -> None:
    """Read the options and publish."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="folder with adapter_model.safetensors")
    parser.add_argument("--public", action="store_true", help="make the repos public (default: private)")
    args = parser.parse_args()
    api = get_api()
    publish_dataset(api, private=not args.public)
    publish_model(api, Path(args.adapter), private=not args.public)


if __name__ == "__main__":
    main()
