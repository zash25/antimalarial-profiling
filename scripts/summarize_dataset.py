"""Summarize a curated molecular activity dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from scripts._utils import update_observation_log, utc_now
except ModuleNotFoundError:
    from _utils import update_observation_log, utc_now


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preview-dir", default="user_outputs/phase_06_eda")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = pd.read_csv(args.input_path)
    label_counts = dataset["activity_label"].value_counts().sort_index().to_dict()
    summary = {
        "generated_at_utc": utc_now(),
        "input_path": args.input_path,
        "molecule_count": int(len(dataset)),
        "unique_assay_count": int(dataset["assay_count"].sum()) if "assay_count" in dataset else None,
        "class_counts": {str(key): int(value) for key, value in label_counts.items()},
        "missing_values": {key: int(value) for key, value in dataset.isna().sum().items()},
        "ic50_nM": {
            key: float(value)
            for key, value in dataset["standard_value_nM"].describe().to_dict().items()
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Summary",
        "",
        f"- Molecules: {summary['molecule_count']}",
        f"- Active (1): {summary['class_counts'].get('1', 0)}",
        f"- Inactive (0): {summary['class_counts'].get('0', 0)}",
        f"- Median IC50: {summary['ic50_nM']['50%']:.2f} nM",
        f"- Summary JSON: `{out_path}`",
    ]
    (preview_dir / "dataset_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    update_observation_log(
        "Phase 6: Exploratory Analysis",
        [
            f"Summarized {summary['molecule_count']} labeled molecules.",
            f"Inspect {preview_dir / 'dataset_summary.md'} before modeling.",
        ],
    )
    print(f"Wrote dataset summary to {out_path}")


if __name__ == "__main__":
    main()
