"""Create the manuscript's side-by-side study and matched-benchmark figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "outputs" / "published_study_benchmark" / "summary.csv"
OUTPUT = ROOT / "comparison_2025_pfdhodh.png"


def main() -> None:
    summary = pd.read_csv(RESULTS, index_col="model")
    colors = {"repo": "#147D75", "paper": "#C45D43", "ink": "#18323A", "muted": "#557079", "line": "#D8E3E3", "pale": "#F3F7F6"}
    fig = plt.figure(figsize=(15.5, 9.2), facecolor="white")
    fig.text(0.055, 0.945, "OVERLAPPING DATA. A NEW EVALUATION QUESTION.", fontsize=24, weight="bold", color=colors["ink"])
    fig.text(0.056, 0.909, "This repository and Iwuala et al. (2025), PfDHODH / ChEMBL CHEMBL3486", fontsize=12.5, color=colors["muted"])

    def panel(x: float, y: float, width: float, height: float, color: str, title: str, lines: list[str]) -> None:
        fig.patches.append(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.008,rounding_size=0.012", transform=fig.transFigure, facecolor="white", edgecolor=colors["line"], linewidth=1.1))
        fig.patches.append(FancyBboxPatch((x, y + height - 0.052), width, 0.052, boxstyle="round,pad=0.008,rounding_size=0.012", transform=fig.transFigure, facecolor=color, edgecolor=color, linewidth=0))
        fig.text(x + 0.018, y + height - 0.035, title, fontsize=13.5, weight="bold", color="white")
        yy = y + height - 0.092
        for line in lines:
            fig.text(x + 0.02, yy, line, fontsize=10.8, color=colors["ink"], va="top", linespacing=1.35)
            yy -= 0.052

    panel(0.055, 0.635, 0.435, 0.235, colors["repo"], "THIS PROJECT: EXACT-MEASUREMENT ANALYSIS", [
        "413 exact records -> 368 unique structures; median IC50 per structure.",
        "Labels: active <=1 uM; all exact values >1 uM inactive (201 / 167).",
        "Morgan RF vs fine-tuned ChemBERTa; five scaffold splits; ROC-AUC 0.648 vs 0.464.",
    ])
    panel(0.51, 0.635, 0.435, 0.235, colors["paper"], "IWUALA ET AL. (2025): BROADER QSAR SEARCH", [
        "602 initial records; 551 curated structures; 465 binary-labeled compounds.",
        "Labels: active <=1 uM; inactive >10 uM; intermediate values excluded.",
        "12 x 12 model search; paper describes Kennard-Stone, notebook code uses random 80/20.",
    ])

    fig.patches.append(FancyBboxPatch((0.055, 0.545), 0.89, 0.064, boxstyle="round,pad=0.008,rounding_size=0.012", transform=fig.transFigure, facecolor="#E8F1F0", edgecolor="none"))
    fig.text(0.075, 0.578, "RECORD AUDIT", fontsize=11.2, weight="bold", color=colors["repo"], va="center")
    fig.text(0.23, 0.578, "281 shared IDs: structures and labels all match  |  181 additional censored-only IDs  |  3 IDs absent from local raw snapshot", fontsize=10.8, color=colors["ink"], va="center")

    fig.text(0.065, 0.49, "MATCHED SCAFFOLD-HELD-OUT REANALYSIS", fontsize=14, weight="bold", color=colors["ink"])
    fig.text(0.065, 0.462, "All three 100-tree random forests use the same 465 compounds, five scaffold partitions, train-only variance filtering and oversampling.", fontsize=10.5, color=colors["muted"])

    axes = fig.add_axes([0.1, 0.18, 0.48, 0.23])
    models = ["PaDEL PubChem RF", "Morgan RF", "PaDEL SubstructureCount RF"]
    metric_names = ["PaDEL PubChem RF", "Morgan RF", "PaDEL SubstructureCount RF"]
    means = [summary.loc[name, "roc_auc_mean"] for name in metric_names]
    stds = [summary.loc[name, "roc_auc_std"] for name in metric_names]
    bars = axes.barh(models[::-1], means[::-1], xerr=stds[::-1], color=[colors["repo"], "#448F88", "#8CB8B3"], capsize=3, height=0.62)
    axes.set_xlim(0, 1.02)
    axes.set_xlabel("ROC-AUC (mean +/- SD across five repeats)", fontsize=9.5, color=colors["ink"])
    axes.tick_params(axis="both", labelsize=9, colors=colors["ink"], length=0)
    axes.grid(axis="x", color=colors["line"], linewidth=0.8)
    axes.set_axisbelow(True)
    for spine in axes.spines.values():
        spine.set_visible(False)
    for bar, val, err in zip(bars, means[::-1], stds[::-1]):
        axes.text(min(val + err + 0.025, 0.96), bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9.2, color=colors["ink"])

    mcc = [summary.loc[name, "mcc_mean"] for name in metric_names]
    mcc_sd = [summary.loc[name, "mcc_std"] for name in metric_names]
    fig.text(0.65, 0.397, "MCC (mean +/- SD)", fontsize=10.5, weight="bold", color=colors["ink"])
    for index, (name, mean, sd) in enumerate(zip(models, mcc, mcc_sd)):
        y = 0.36 - index * 0.052
        fig.text(0.65, y, name, fontsize=9.8, color=colors["muted"])
        fig.text(0.92, y, f"{mean:.3f} +/- {sd:.3f}", fontsize=9.8, color=colors["ink"], ha="right")

    fig.patches.append(FancyBboxPatch((0.055, 0.065), 0.89, 0.055, boxstyle="round,pad=0.008,rounding_size=0.012", transform=fig.transFigure, facecolor=colors["ink"], edgecolor=colors["ink"]))
    fig.text(0.075, 0.092, "TAKEAWAY", fontsize=10.5, weight="bold", color="#9BD4CA", va="center")
    fig.text(0.185, 0.092, "The matched run favors PubChem fingerprints on average, but repeat variability is material; this is not prospective or wet-lab validation.", fontsize=10.5, color="white", va="center")
    fig.text(0.056, 0.027, "Source: Iwuala et al. (2025), DOI 10.1016/j.insi.2025.100019; local ChEMBL snapshot. Error bars are split SD, not confidence intervals.", fontsize=8.5, color=colors["muted"])
    fig.savefig(OUTPUT, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
