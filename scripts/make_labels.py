"""Create binary activity labels from curated IC50 data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from scripts._utils import update_observation_log
except ModuleNotFoundError:
    from _utils import update_observation_log


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, help="Clean activity CSV.")
    parser.add_argument("--out", required=True, help="Labeled CSV output.")
    parser.add_argument("--threshold-nm", type=float, default=1000.0, help="Active IC50 threshold in nM.")
    parser.add_argument(
        "--preview-dir", default="user_outputs/phase_05_data_curation", help="Visible output folder."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.threshold_nm <= 0:
        raise ValueError("--threshold-nm must be greater than zero.")
    dataset = pd.read_csv(args.input_path)
    if "standard_value_nM" not in dataset:
        raise ValueError("Input must contain standard_value_nM.")
    dataset["activity_label"] = (dataset["standard_value_nM"] <= args.threshold_nm).astype(int)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(out_path, index=False)

    counts = dataset["activity_label"].value_counts().rename_axis("activity_label").reset_index(name="count")
    counts["label_name"] = counts["activity_label"].map({1: "active", 0: "inactive"})
    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    counts.to_csv(preview_dir / "label_distribution.csv", index=False)
    update_observation_log(
        "Phase 5: Activity Labels",
        [
            f"Created active/inactive labels using IC50 <= {args.threshold_nm:g} nM.",
            f"Label distribution: {preview_dir / 'label_distribution.csv'}.",
        ],
    )
    print(f"Wrote {len(dataset)} labeled molecules to {out_path}")


if __name__ == "__main__":
    main()
