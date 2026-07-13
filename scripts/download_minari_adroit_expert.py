"""Download the Minari D4RL Adroit expert datasets used by RoboBase."""

from __future__ import annotations

import argparse


ADROIT_EXPERT_DATASETS = (
    "D4RL/pen/expert-v2",
    "D4RL/door/expert-v2",
    "D4RL/hammer/expert-v2",
    "D4RL/relocate/expert-v2",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ADROIT_EXPERT_DATASETS,
        choices=ADROIT_EXPERT_DATASETS,
        help="Subset of supported Minari D4RL Adroit expert datasets to download.",
    )
    args = parser.parse_args()

    try:
        import minari
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency `minari`. Install the d4rl extra or run "
            "`uv pip install --python .venv/bin/python 'minari[hf,gcs]==0.5.3'`."
        ) from exc

    for dataset_id in args.datasets:
        print(f"Downloading {dataset_id}")
        minari.download_dataset(dataset_id)


if __name__ == "__main__":
    main()
