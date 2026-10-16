"""Check whether Morgan random-forest results depend on the activity cutoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.split_utils import indices_from_splits, scaffold_groups
from train.train_baseline import fingerprint_matrix, metric_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Deduplicated IC50 CSV before binary labeling.")
    parser.add_argument("--split-dir", default="outputs/robust_evaluation/splits")
    parser.add_argument("--out", default="outputs/threshold_sensitivity")
    parser.add_argument("--preview-dir", default="user_outputs/phase_12_final_strengthening")
    parser.add_argument("--thresholds-nm", nargs="+", type=float, default=[500.0, 1000.0, 2000.0])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    source_path = Path(args.data)
    dataset = pd.read_csv(source_path)
    required = {"canonical_smiles", "standard_value_nM"}
    if not required.issubset(dataset.columns):
        raise ValueError(f"Input data must include columns: {sorted(required)}")
    if dataset["canonical_smiles"].duplicated().any():
        raise ValueError("Input must contain one row per canonical SMILES.")
    if not args.thresholds_nm or any(value <= 0 for value in args.thresholds_nm):
        raise ValueError("All activity cutoffs must be greater than zero.")

    split_files = sorted(Path(args.split_dir).glob("repeat_*_split_ids.csv"))
    if len(split_files) < 2:
        raise ValueError("At least two saved scaffold split files are required.")

    groups = scaffold_groups(dataset)
    features = fingerprint_matrix(dataset["canonical_smiles"].astype(str).tolist())
    rows: list[dict[str, int | float]] = []
    for threshold in args.thresholds_nm:
        labeled = dataset.copy()
        labeled["activity_label"] = (labeled["standard_value_nM"] <= threshold).astype(int)
        for repeat_index, split_path in enumerate(split_files):
            indices = indices_from_splits(labeled, split_path)
            scaffold_sets = {name: set(groups[selected]) for name, selected in indices.items()}
            if any(
                scaffold_sets[left] & scaffold_sets[right]
                for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
            ):
                raise ValueError(f"Scaffold overlap found in {split_path}.")

            seed = 42 + repeat_index
            model = RandomForestClassifier(
                n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=seed
            )
            model.fit(features[indices["train"]], labeled["activity_label"].to_numpy()[indices["train"]])
            test_indices = indices["test"]
            probability = model.predict_proba(features[test_indices])[:, 1]
            metrics = metric_payload(labeled["activity_label"].to_numpy()[test_indices], probability)
            rows.append(
                {
                    "threshold_nM": threshold,
                    "repeat": repeat_index + 1,
                    "seed": seed,
                    "active_molecules": int(labeled["activity_label"].sum()),
                    "inactive_molecules": int((labeled["activity_label"] == 0).sum()),
                    "test_molecules": len(test_indices),
                    **metrics,
                }
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(out_dir / "metrics_by_repeat.csv", index=False)
    summary_rows = []
    for threshold, frame in metrics_frame.groupby("threshold_nM", sort=True):
        summary_row: dict[str, int | float] = {
            "threshold_nM": float(threshold),
            "active_molecules": int(frame["active_molecules"].iloc[0]),
            "inactive_molecules": int(frame["inactive_molecules"].iloc[0]),
            "repeats": len(frame),
        }
        for metric in ("roc_auc", "pr_auc", "f1", "precision", "recall"):
            summary_row[f"{metric}_mean"] = float(frame[metric].mean())
            summary_row[f"{metric}_sd"] = float(frame[metric].std(ddof=1))
        summary_rows.append(summary_row)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(out_dir / "summary.csv", index=False)

    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    lines = [
        "# Activity-Cutoff Sensitivity Check",
        "",
        f"- Dataset molecules: {len(dataset)}",
        f"- Scaffold split repeats: {len(split_files)}; saved 1,000 nM split assignments reused at every cutoff.",
        "- Model: 500-tree Morgan radius-2, 2,048-bit random forest with balanced class weights.",
        "- Metrics are mean +/- sample standard deviation across repeated splits; this is split variability, not a confidence interval.",
        f"- Clean dataset SHA-256: `{digest}`",
        "- This is a cutoff sensitivity check for the stronger primary model, not an independent validation dataset.",
        "",
        "| Active cutoff (nM) | Active / inactive | ROC-AUC | PR-AUC | F1 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['threshold_nM']:g} | {row['active_molecules']} / {row['inactive_molecules']} | "
            f"{row['roc_auc_mean']:.3f} +/- {row['roc_auc_sd']:.3f} | "
            f"{row['pr_auc_mean']:.3f} +/- {row['pr_auc_sd']:.3f} | "
            f"{row['f1_mean']:.3f} +/- {row['f1_sd']:.3f} |"
        )
    report = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(report, encoding="utf-8")

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("summary.md", "summary.csv", "metrics_by_repeat.csv"):
        shutil.copy2(out_dir / filename, preview_dir / filename)
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "data_path": str(source_path),
                "data_sha256": digest,
                "thresholds_nM": args.thresholds_nm,
                "split_files": [str(path) for path in split_files],
                "model": {"name": "Morgan random forest", "radius": 2, "bits": 2048, "trees": 500},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(out_dir / "run_metadata.json", preview_dir / "run_metadata.json")
    print(f"Wrote cutoff sensitivity results to {out_dir}")


if __name__ == "__main__":
    main()
