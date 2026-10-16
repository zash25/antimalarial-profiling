"""Run repeated, shared Bemis-Murcko scaffold holdouts for both models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib").resolve()))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._utils import update_observation_log, utc_now
from train.split_utils import scaffold_split_indices, write_split_file


METRICS = ("roc_auc", "pr_auc", "f1", "precision", "recall")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="One-row-per-molecule labeled CSV.")
    parser.add_argument("--out", default="outputs/robust_evaluation")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chemberta-epochs", type=int, default=3)
    parser.add_argument("--skip-chemberta", action="store_true")
    parser.add_argument("--model-name", default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--metadata", default="data/target_metadata.json")
    parser.add_argument("--target-chembl-id")
    parser.add_argument("--standard-type", default="IC50")
    parser.add_argument("--threshold-nm", type=float)
    parser.add_argument("--preview-dir", default="user_outputs/phase_11_robust_evaluation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.repeats < 2:
        raise ValueError("Use at least two scaffold repeats; the default is five.")
    dataset = pd.read_csv(args.data)
    if dataset["canonical_smiles"].duplicated().any():
        raise ValueError("Scaffold evaluation requires one row per canonical SMILES.")
    out_dir = Path(args.out)
    split_dir = out_dir / "splits"
    preview_dir = Path(args.preview_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    from train.fine_tune_chemberta import main as chemberta_main
    from train.train_baseline import main as baseline_main

    split_summaries: list[dict[str, int | float]] = []
    metric_rows: list[dict[str, int | float | str]] = []
    model_specs = [("morgan_random_forest", "Morgan random forest")]
    if not args.skip_chemberta:
        model_specs.append(("chemberta", "ChemBERTa"))

    for repeat in range(args.repeats):
        repeat_seed = args.seed + repeat
        indices, groups = scaffold_split_indices(dataset, repeat_seed)
        split_path = split_dir / f"repeat_{repeat + 1:02d}_split_ids.csv"
        write_split_file(dataset, indices, split_path)
        for name, selected in indices.items():
            labels = dataset.iloc[selected]["activity_label"]
            split_summaries.append(
                {
                    "repeat": repeat + 1,
                    "seed": repeat_seed,
                    "split": name,
                    "molecule_count": len(selected),
                    "active_count": int((labels == 1).sum()),
                    "inactive_count": int((labels == 0).sum()),
                    "scaffold_count": len(set(groups[selected])),
                }
            )

        for model_key, model_name in model_specs:
            model_out = out_dir / f"repeat_{repeat + 1:02d}" / model_key
            common = ["--data", args.data, "--out", str(model_out), "--seed", str(repeat_seed), "--split-ids", str(split_path), "--no-preview"]
            if model_key == "morgan_random_forest":
                baseline_main(common)
            else:
                chemberta_main(
                    common
                    + [
                        "--model-name", args.model_name,
                        "--epochs", str(args.chemberta_epochs),
                        "--device", args.device,
                        "--no-save-model",
                    ]
                )
            payload = json.loads((model_out / "metrics.json").read_text(encoding="utf-8"))
            row: dict[str, int | float | str] = {
                "repeat": repeat + 1,
                "seed": repeat_seed,
                "model": model_name,
                "test_molecules": int(payload["split_sizes"]["test"]),
            }
            row.update({metric: float(payload["test"][metric]) for metric in METRICS})
            metric_rows.append(row)
        print(f"Completed scaffold evaluation repeat {repeat + 1}/{args.repeats}")

    pd.DataFrame(split_summaries).to_csv(out_dir / "split_summary.csv", index=False)
    run_metrics = pd.DataFrame(metric_rows)
    run_metrics.to_csv(out_dir / "metrics_by_repeat.csv", index=False)
    models_summary: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, group in run_metrics.groupby("model", sort=False):
        models_summary[model_name] = {
            metric: {
                "mean": float(group[metric].mean()),
                "std": float(group[metric].std(ddof=1)),
                "values": [float(value) for value in group[metric]],
            }
            for metric in METRICS
        }
    summary = {
        "generated_at_utc": utc_now(),
        "split_strategy": "Repeated Bemis-Murcko scaffold holdout; scaffold groups are disjoint across partitions.",
        "repeats": args.repeats,
        "base_seed": args.seed,
        "dataset_size": len(dataset),
        "dataset": {
            "path": str(Path(args.data)),
            "sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "standard_type": args.standard_type,
            "activity_label_threshold_nM": args.threshold_nm,
            "classifier_probability_threshold": 0.5,
        },
        "target": {
            "target_chembl_id": args.target_chembl_id,
            "metadata_path": args.metadata,
        },
        "model_settings": {
            "morgan_random_forest": {"fingerprint": "Morgan", "radius": 2, "bits": 2048, "trees": 500, "class_weight": "balanced"},
            "chemberta": None if args.skip_chemberta else {"model_name": args.model_name, "epochs": args.chemberta_epochs, "device": args.device},
        },
        "models": models_summary,
    }
    metadata_path = Path(args.metadata)
    if metadata_path.exists():
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected_target = metadata_payload.get("selected_target") or {}
        summary["target"].update(
            {
                "target_chembl_id": args.target_chembl_id or selected_target.get("target_chembl_id"),
                "query": metadata_payload.get("query"),
                "organism": metadata_payload.get("organism"),
                "retrieved_at_utc": metadata_payload.get("created_at_utc"),
            }
        )
    elif args.target_chembl_id:
        summary["target"]["target_chembl_id"] = args.target_chembl_id
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    report_lines = [
        "# Repeated Scaffold Evaluation",
        "",
        f"- Dataset molecules: {len(dataset)}",
        f"- Repeats: {args.repeats}",
        "- Split: Bemis-Murcko scaffold groups, kept disjoint across train, validation, and test.",
        "- Reported values: mean +/- sample standard deviation across repeats.",
        "- These repeated splits share molecules across repeats; the standard deviation describes split sensitivity and is not a confidence interval.",
        f"- Target: {summary['target'].get('target_chembl_id') or 'not specified'}; activity: {args.standard_type}; active threshold: {args.threshold_nm if args.threshold_nm is not None else 'read from input'} nM.",
        f"- Labeled dataset SHA-256: `{summary['dataset']['sha256']}`",
        "",
        "| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, model_metrics in models_summary.items():
        vals = [f"{model_metrics[key]['mean']:.3f} +/- {model_metrics[key]['std']:.3f}" for key in METRICS]
        report_lines.append(f"| {model_name} | " + " | ".join(vals) + " |")
    report_path = out_dir / "summary.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    metric_names = ["roc_auc", "pr_auc", "f1"]
    model_names = list(models_summary)
    x = np.arange(len(metric_names))
    width = 0.75 / max(1, len(model_names))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for index, model_name in enumerate(model_names):
        means = [models_summary[model_name][metric]["mean"] for metric in metric_names]
        errors = [models_summary[model_name][metric]["std"] for metric in metric_names]
        axis.bar(x - 0.375 + width / 2 + index * width, means, width, yerr=errors, capsize=3, label=model_name)
    axis.set(xticks=x, xticklabels=["ROC-AUC", "PR-AUC", "F1"], ylim=(0, 1.08), ylabel="Score", title="Repeated scaffold holdout performance")
    axis.legend()
    figure.tight_layout()
    plot_path = out_dir / "scaffold_model_comparison.png"
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)

    for filename in ("summary.md", "summary.json", "metrics_by_repeat.csv", "split_summary.csv", "scaffold_model_comparison.png"):
        shutil.copy2(out_dir / filename, preview_dir / filename)
    update_observation_log(
        "Phase 11: Robust Scaffold Evaluation",
        [
            f"Evaluated {len(model_specs)} model(s) over {args.repeats} scaffold-disjoint repeats.",
            f"Inspect {preview_dir / 'summary.md'} and scaffold_model_comparison.png.",
            "Each repeat uses the same molecule split for both models; per-repeat metrics and split composition are saved alongside the summary.",
        ],
    )
    print(f"Wrote repeated scaffold evaluation to {out_dir}")


if __name__ == "__main__":
    main()
