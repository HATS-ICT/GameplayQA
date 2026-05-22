"""Download the GameplayQA dataset from the Hugging Face Hub into a local folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "wangyz1999/GameplayQA"


def download(destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {REPO_ID} -> {destination} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(destination),
    )
    print(f"Done. Dataset is in {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--destination",
        type=Path,
        default=Path("data"),
        help="Local folder for the dataset (default: ./data)",
    )
    args = parser.parse_args()
    download(args.destination)


if __name__ == "__main__":
    main()
