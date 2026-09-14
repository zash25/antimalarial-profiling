"""Create compact, publication-ready exploratory plots for a labeled dataset."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from scripts._utils import update_observation_log
except ModuleNotFoundError:
    from _utils import update_observation_log


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", required=True, help="Figure directory.")
    parser.add_argument("--preview-dir", default="user_outputs/phase_06_eda")
    return parser.parse_args(argv)


def save_figure(figure: plt.Figure, path: Path, preview_dir: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    shutil.copy2(path, preview_dir / path.name)
    plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = pd.read_csv(args.input_path)
    out_dir = Path(args.out)
    preview_dir = Path(args.preview_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(dataset["pIC50"], bins=24, color="#1677a8", edgecolor="white")
    axis.set(xlabel="pIC50", ylabel="Molecule count", title="PfDHODH potency distribution")
    save_figure(figure, out_dir / "pic50_distribution.png", preview_dir)

    counts = dataset["activity_label"].value_counts().reindex([0, 1], fill_value=0)
    figure, axis = plt.subplots(figsize=(5, 4))
    bars = axis.bar(["Inactive", "Active"], counts.tolist(), color=["#d95f5f", "#37a866"])
    axis.set(ylabel="Molecule count", title="Activity class balance")
    axis.bar_label(bars, padding=3)
    save_figure(figure, out_dir / "class_balance.png", preview_dir)

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(dataset["assay_count"], bins=range(1, int(dataset["assay_count"].max()) + 2), align="left", color="#7b5ea7", edgecolor="white")
    axis.set(xlabel="Assays contributing to molecule", ylabel="Molecule count", title="Assay support per molecule")
    save_figure(figure, out_dir / "assay_support.png", preview_dir)

    update_observation_log(
        "Phase 6: Dataset Plots",
        [
            "Created pIC50 distribution, class-balance, and assay-support figures.",
            f"Inspect PNG files in {preview_dir}.",
        ],
    )
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
