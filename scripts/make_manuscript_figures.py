"""Compare saved model metrics and create final reporting artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scripts._utils import update_observation_log
except ModuleNotFoundError:
    from _utils import update_observation_log


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", default="outputs/baselines/random_forest/metrics.json")
    parser.add_argument("--chemberta-metrics", default="outputs/chemberta_target/metrics.json")
    parser.add_argument("--out", required=True, help="Figure/report directory.")
    parser.add_argument("--preview-dir", default="user_outputs/phase_09_manuscript_outputs")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sources = [("Morgan RF", Path(args.baseline_metrics)), ("ChemBERTa", Path(args.chemberta_metrics))]
    available = []
    for name, path in sources:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            available.append((name, payload["test"]))
    if not available:
        raise FileNotFoundError("No model metrics were found. Train at least one model first.")
    out_dir, preview_dir = Path(args.out), Path(args.preview_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    metric_names = ["roc_auc", "pr_auc", "f1"]
    figure, axis = plt.subplots(figsize=(7, 4))
    positions, width = list(range(len(metric_names))), 0.7 / len(available)
    for index, (name, metrics) in enumerate(available):
        offsets = [position - 0.35 + width / 2 + index * width for position in positions]
        axis.bar(offsets, [metrics[metric] for metric in metric_names], width=width, label=name)
    axis.set(xticks=positions, xticklabels=["ROC-AUC", "PR-AUC", "F1"], ylim=(0, 1.05), ylabel="Score", title="Held-out test performance")
    axis.legend()
    figure.tight_layout()
    figure_path = out_dir / "model_comparison.png"
    figure.savefig(figure_path, dpi=220)
    plt.close(figure)
    lines = ["# Final Model Results", ""]
    for name, metrics in available:
        lines.append(f"## {name}")
        lines.extend(f"- {metric.upper()}: {metrics[metric]:.3f}" for metric in metric_names)
        lines.append("")
    report_path = out_dir / "manuscript_results.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(figure_path, preview_dir / figure_path.name)
    shutil.copy2(report_path, preview_dir / report_path.name)
    update_observation_log("Phase 9: Evaluation and Manuscript Outputs", [f"Created a held-out test metric comparison for {len(available)} model(s).", f"Inspect {preview_dir / 'model_comparison.png'} and manuscript_results.md."])
    print(f"Wrote final reporting artifacts to {out_dir}")


if __name__ == "__main__":
    main()
