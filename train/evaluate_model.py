"""Evaluate a trained model using its saved held-out test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Labeled dataset used for training.")
    parser.add_argument("--model", required=True, help="Model output directory.")
    parser.add_argument("--out", required=True, help="Evaluation JSON output path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model_dir = Path(args.model)
    predictions = pd.read_csv(model_dir / "test_predictions.csv")
    if predictions["activity_label"].nunique() != 2:
        raise ValueError("Held-out test predictions must contain both classes.")
    labels = predictions["activity_label"].to_numpy(dtype=int)
    probability = predictions["active_probability"].to_numpy(dtype=float)
    predicted = (probability >= 0.5).astype(int)
    report = {"model_dir": str(model_dir), "test_rows": int(len(predictions)), "accuracy": float(accuracy_score(labels, predicted)), "roc_auc": float(roc_auc_score(labels, probability)), "pr_auc": float(average_precision_score(labels, probability)), "f1": float(f1_score(labels, predicted)), "precision": float(precision_score(labels, predicted, zero_division=0)), "recall": float(recall_score(labels, predicted, zero_division=0))}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Evaluation ROC-AUC: {report['roc_auc']:.3f}")
    print(f"Wrote evaluation to {out_path}")


if __name__ == "__main__":
    main()
