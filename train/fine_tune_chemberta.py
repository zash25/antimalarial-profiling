"""Fine-tune ChemBERTa for binary activity prediction from SMILES."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib").resolve()))

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._utils import update_observation_log, utc_now


class SmilesDataset(Dataset):
    def __init__(self, smiles: list[str], labels: np.ndarray, tokenizer: object, max_length: int) -> None:
        self.encodings = tokenizer(smiles, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return item


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--text-column", default="canonical_smiles")
    parser.add_argument("--label-column", default="activity_label")
    parser.add_argument("--model-name", default="DeepChem/ChemBERTa-77M-MLM")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probabilities(model: object, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probability_items, label_items = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            probability_items.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            label_items.extend(labels.cpu().numpy())
    return np.asarray(label_items), np.asarray(probability_items)


def metric_payload(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "f1": float(f1_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    set_seed(args.seed)
    dataset = pd.read_csv(args.data).dropna(subset=[args.text_column, args.label_column]).reset_index(drop=True)
    labels = dataset[args.label_column].to_numpy(dtype=int)
    if len(np.unique(labels)) != 2:
        raise ValueError("ChemBERTa fine-tuning needs both active and inactive labels.")
    train_val_idx, test_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=args.seed, stratify=labels)
    train_idx, validation_idx = train_test_split(train_val_idx, test_size=0.25, random_state=args.seed, stratify=labels[train_val_idx])
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)
    text = dataset[args.text_column].astype(str).tolist()
    train_loader = DataLoader(SmilesDataset([text[index] for index in train_idx], labels[train_idx], tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(SmilesDataset([text[index] for index in validation_idx], labels[validation_idx], tokenizer, args.max_length), batch_size=args.batch_size)
    test_loader = DataLoader(SmilesDataset([text[index] for index in test_idx], labels[test_idx], tokenizer, args.max_length), batch_size=args.batch_size)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress: list[dict[str, float | int]] = []
    best_validation = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation_labels, validation_probability = probabilities(model, validation_loader, device)
        validation_metrics = metric_payload(validation_labels, validation_probability)
        progress.append({"epoch": epoch, "train_loss": float(np.mean(losses)), **validation_metrics})
        if validation_metrics["roc_auc"] > best_validation:
            best_validation = validation_metrics["roc_auc"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    best_model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    test_labels, test_probability = probabilities(best_model, test_loader, device)
    metrics = {
        "model_type": "chemberta_binary_classifier", "model_name": args.model_name,
        "generated_at_utc": utc_now(), "seed": args.seed, "device": str(device),
        "dataset_size": int(len(dataset)),
        "split_sizes": {"train": int(len(train_idx)), "validation": int(len(validation_idx)), "test": int(len(test_idx))},
        "best_validation_roc_auc": best_validation, "test": metric_payload(test_labels, test_probability),
    }
    pd.DataFrame(progress).to_csv(out_dir / "training_progress.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "model_metadata.json").write_text(json.dumps({"model_type": "chemberta_binary_classifier", "text_column": args.text_column, "label_column": args.label_column, "max_length": args.max_length, "seed": args.seed}, indent=2), encoding="utf-8")
    split_records = []
    for name, indices in [("train", train_idx), ("validation", validation_idx), ("test", test_idx)]:
        split_records.extend({"canonical_smiles": dataset.iloc[index][args.text_column], "split": name} for index in indices)
    pd.DataFrame(split_records).to_csv(out_dir / "split_ids.csv", index=False)
    pd.DataFrame({"canonical_smiles": dataset.iloc[test_idx][args.text_column], "activity_label": test_labels, "active_probability": test_probability, "predicted_label": (test_probability >= 0.5).astype(int)}).to_csv(out_dir / "test_predictions.csv", index=False)

    figure, axis = plt.subplots(figsize=(6, 4))
    progress_frame = pd.DataFrame(progress)
    axis.plot(progress_frame["epoch"], progress_frame["train_loss"], marker="o", color="#1677a8")
    axis.set(xlabel="Epoch", ylabel="Training loss", title="ChemBERTa training progress")
    figure.tight_layout()
    figure.savefig(out_dir / "training_loss.png", dpi=180)
    plt.close(figure)
    preview_dir = Path("user_outputs/phase_08_chemberta")
    preview_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_dir / "training_progress.csv", preview_dir / "training_progress.csv")
    shutil.copy2(out_dir / "training_loss.png", preview_dir / "training_loss.png")
    (preview_dir / "run_summary.md").write_text("\n".join(["# ChemBERTa Run Summary", "", f"- Device: {device}", f"- Epochs: {args.epochs}", f"- Best validation ROC-AUC: {best_validation:.3f}", f"- Test ROC-AUC: {metrics['test']['roc_auc']:.3f}", f"- Test PR-AUC: {metrics['test']['pr_auc']:.3f}"]) + "\n", encoding="utf-8")
    update_observation_log("Phase 8: ChemBERTa Fine-Tuning", [f"Fine-tuned {args.model_name} for {args.epochs} epoch(s) on {device}.", f"Test ROC-AUC: {metrics['test']['roc_auc']:.3f}; PR-AUC: {metrics['test']['pr_auc']:.3f}.", f"Inspect {preview_dir / 'run_summary.md'} and training_loss.png."])
    print(f"ChemBERTa test ROC-AUC: {metrics['test']['roc_auc']:.3f}")
    print(f"Wrote model and metrics to {out_dir}")


if __name__ == "__main__":
    main()
