"""Evaluate a conservative use of right-censored IC50 values as negatives."""

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

from scripts.preprocess_smiles import UNIT_TO_NM, canonicalize_smiles
from train.split_utils import scaffold_groups, scaffold_split_indices, write_split_file
from train.train_baseline import fingerprint_matrix, metric_payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-data", default="data/processed/chembl3486_ic50_clean.csv")
    parser.add_argument("--censored-data", default="data/processed/chembl3486_ic50_rejected_rows.csv")
    parser.add_argument("--threshold-nm", type=float, default=1000.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="outputs/censored_negative_sensitivity")
    parser.add_argument("--preview-dir", default="user_outputs/phase_12_final_strengthening")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.threshold_nm <= 0 or args.repeats < 2:
        raise ValueError("Use a positive activity cutoff and at least two repeats.")

    clean_path = Path(args.clean_data)
    censored_path = Path(args.censored_data)
    exact = pd.read_csv(clean_path)
    censored = pd.read_csv(censored_path, dtype=str).fillna("")
    exact["activity_label"] = (exact["standard_value_nM"] <= args.threshold_nm).astype(int)
    exact["activity_source"] = "exact_measurement"
    exact_structures = set(exact["canonical_smiles"].astype(str))

    censored["bound_nM"] = pd.to_numeric(censored["standard_value"], errors="coerce") * censored[
        "standard_units"
    ].str.upper().str.strip().map(UNIT_TO_NM)
    eligible = censored.loc[
        (censored["standard_relation"].str.strip() == ">")
        & (censored["bound_nM"] > args.threshold_nm)
    ].copy()
    eligible["canonical_smiles"] = eligible["canonical_smiles"].map(canonicalize_smiles)
    eligible = eligible.dropna(subset=["canonical_smiles"])
    eligible = eligible.loc[~eligible["canonical_smiles"].isin(exact_structures)]
    eligible = eligible.drop_duplicates("canonical_smiles")

    added = pd.DataFrame(
        {
            "canonical_smiles": eligible["canonical_smiles"].astype(str),
            "standard_value_nM": eligible["bound_nM"].astype(float),
            "activity_label": 0,
            "activity_source": "right_censored_lower_bound_above_cutoff",
        }
    )
    dataset = pd.concat([exact, added], ignore_index=True, sort=False)
    if dataset["canonical_smiles"].duplicated().any():
        raise ValueError("Censoring sensitivity input contains duplicate structures.")

    groups = scaffold_groups(dataset)
    features = fingerprint_matrix(dataset["canonical_smiles"].astype(str).tolist())
    rows: list[dict[str, int | float]] = []
    split_rows: list[dict[str, int | float | str]] = []
    out_dir = Path(args.out)
    split_dir = out_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    labels = dataset["activity_label"].to_numpy(dtype=int)

    for repeat in range(args.repeats):
        seed = args.seed + repeat
        indices, split_groups = scaffold_split_indices(dataset, seed)
        split_path = split_dir / f"repeat_{repeat + 1:02d}_split_ids.csv"
        write_split_file(dataset, indices, split_path)
        group_sets = {name: set(split_groups[selected]) for name, selected in indices.items()}
        if any(
            group_sets[left] & group_sets[right]
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ):
            raise ValueError(f"Scaffold overlap found in repeat {repeat + 1}.")

        for name, selected in indices.items():
            selected_labels = labels[selected]
            split_rows.append(
                {
                    "repeat": repeat + 1,
                    "seed": seed,
                    "split": name,
                    "molecules": len(selected),
                    "active": int(selected_labels.sum()),
                    "inactive": int((selected_labels == 0).sum()),
                    "scaffolds": len(set(groups[selected])),
                }
            )

        model = RandomForestClassifier(
            n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=seed
        )
        model.fit(features[indices["train"]], labels[indices["train"]])
        test_indices = indices["test"]
        probabilities = model.predict_proba(features[test_indices])[:, 1]
        rows.append(
            {
                "repeat": repeat + 1,
                "seed": seed,
                "test_molecules": len(test_indices),
                **metric_payload(labels[test_indices], probabilities),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    metric_frame = pd.DataFrame(rows)
    metric_frame.to_csv(out_dir / "metrics_by_repeat.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out_dir / "split_summary.csv", index=False)
    dataset.to_csv(out_dir / "augmented_labeled_dataset.csv", index=False)
    digest = hashlib.sha256((out_dir / "augmented_labeled_dataset.csv").read_bytes()).hexdigest()

    lines = [
        "# Right-Censored Negative Sensitivity Check",
        "",
        f"- Exact-measurement molecules: {len(exact)} ({int(exact['activity_label'].sum())} active; {int((exact['activity_label'] == 0).sum())} inactive).",
        f"- Added censored-only molecules: {len(added)}; each has a reported lower bound above {args.threshold_nm:g} nM and was labeled inactive.",
        f"- Augmented molecules: {len(dataset)}; cutoff: {args.threshold_nm:g} nM.",
        f"- Scaffold-disjoint repeats: {args.repeats}; seeds: {args.seed} through {args.seed + args.repeats - 1}.",
        "- Model: 500-tree Morgan radius-2, 2,048-bit random forest with balanced class weights.",
        "- This is a secondary curation sensitivity analysis, not external validation. The one censored structure also present in the exact dataset was not added or used to override its exact-measurement label.",
        f"- Augmented dataset SHA-256: `{digest}`",
        "",
        "| Metric | Mean +/- sample SD |",
        "| --- | ---: |",
    ]
    for metric in ("roc_auc", "pr_auc", "f1", "precision", "recall"):
        lines.append(f"| {metric.upper()} | {metric_frame[metric].mean():.3f} +/- {metric_frame[metric].std(ddof=1):.3f} |")
    report = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "threshold_nM": args.threshold_nm,
                "repeats": args.repeats,
                "seed": args.seed,
                "clean_data": str(clean_path),
                "censored_data": str(censored_path),
                "augmented_data_sha256": digest,
                "censored_records_considered": len(censored),
                "eligible_unique_structures_added": len(added),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("summary.md", "metrics_by_repeat.csv", "split_summary.csv"):
        shutil.copy2(out_dir / filename, preview_dir / f"censored_{filename}")
    print(f"Wrote censored-negative sensitivity results to {out_dir}")


if __name__ == "__main__":
    main()
