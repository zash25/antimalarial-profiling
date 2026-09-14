"""Train a reproducible Morgan fingerprint random-forest baseline."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib").resolve()))

import joblib
import matplotlib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._utils import update_observation_log, utc_now


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--features", default="morgan", choices=["morgan"])
    parser.add_argument("--model", default="random_forest", choices=["random_forest"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def fingerprint_matrix(smiles_values: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    features = []
    for smiles in smiles_values:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES in model dataset: {smiles}")
        fingerprint = generator.GetFingerprint(molecule)
        vector = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, vector)
        features.append(vector)
    return np.vstack(features)


def metric_payload(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "f1": float(f1_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = pd.read_csv(args.data)
    labels = dataset["activity_label"].to_numpy(dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("Baseline training needs both active and inactive labels.")
    train_val_idx, test_idx = train_test_split(
        np.arange(len(dataset)), test_size=0.2, random_state=args.seed, stratify=labels
    )
    train_idx, validation_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=args.seed,
        stratify=labels[train_val_idx],
    )
    features = fingerprint_matrix(dataset["canonical_smiles"].tolist())
    classifier = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=args.seed
    )
    classifier.fit(features[train_idx], labels[train_idx])
    validation_probability = classifier.predict_proba(features[validation_idx])[:, 1]
    test_probability = classifier.predict_proba(features[test_idx])[:, 1]
    metrics = {
        "model_type": "morgan_random_forest",
        "generated_at_utc": utc_now(),
        "seed": args.seed,
        "dataset_size": int(len(dataset)),
        "split_sizes": {"train": int(len(train_idx)), "validation": int(len(validation_idx)), "test": int(len(test_idx))},
        "validation": metric_payload(labels[validation_idx], validation_probability),
        "test": metric_payload(labels[test_idx], test_probability),
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "model_metadata.json").write_text(
        json.dumps({"model_type": "morgan_random_forest", "radius": 2, "fp_size": 2048, "seed": args.seed}, indent=2),
        encoding="utf-8",
    )
    split_records = []
    for name, indices in [("train", train_idx), ("validation", validation_idx), ("test", test_idx)]:
        split_records.extend({"canonical_smiles": dataset.iloc[index]["canonical_smiles"], "split": name} for index in indices)
    pd.DataFrame(split_records).to_csv(out_dir / "split_ids.csv", index=False)
    pd.DataFrame({
        "canonical_smiles": dataset.iloc[test_idx]["canonical_smiles"],
        "activity_label": labels[test_idx],
        "active_probability": test_probability,
        "predicted_label": (test_probability >= 0.5).astype(int),
    }).to_csv(out_dir / "test_predictions.csv", index=False)

    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(labels[test_idx], (test_probability >= 0.5).astype(int), ax=axis, colorbar=False)
    axis.set_title("Random forest test confusion matrix")
    figure.tight_layout()
    figure.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(labels[test_idx], test_probability, ax=axis)
    axis.set_title("Random forest test ROC curve")
    figure.tight_layout()
    figure.savefig(out_dir / "roc_curve.png", dpi=180)
    plt.close(figure)

    preview_dir = Path("user_outputs/phase_07_baselines")
    preview_dir.mkdir(parents=True, exist_ok=True)
    for name in ["confusion_matrix.png", "roc_curve.png"]:
        shutil.copy2(out_dir / name, preview_dir / name)
    summary = [
        "# Baseline Metrics",
        "",
        f"- Test ROC-AUC: {metrics['test']['roc_auc']:.3f}",
        f"- Test PR-AUC: {metrics['test']['pr_auc']:.3f}",
        f"- Test F1: {metrics['test']['f1']:.3f}",
        f"- Test accuracy: {metrics['test']['accuracy']:.3f}",
        f"- Split sizes: {metrics['split_sizes']}",
    ]
    (preview_dir / "baseline_metrics.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    update_observation_log(
        "Phase 7: Random Forest Baseline",
        [
            f"Trained a 2048-bit Morgan fingerprint random forest with seed {args.seed}.",
            f"Test ROC-AUC: {metrics['test']['roc_auc']:.3f}; PR-AUC: {metrics['test']['pr_auc']:.3f}.",
            f"Inspect {preview_dir / 'baseline_metrics.md'} and its diagnostic plots.",
        ],
    )
    print(f"Baseline test ROC-AUC: {metrics['test']['roc_auc']:.3f}")
    print(f"Wrote model and metrics to {out_dir}")


if __name__ == "__main__":
    main()
