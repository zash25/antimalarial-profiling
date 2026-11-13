"""Matched scaffold benchmark against the published PfDHODH QSAR dataset.

The reference CSV and PaDEL XML definitions are external inputs. This script
does not redistribute either third-party artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.split_utils import scaffold_split_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-data", required=True, help="Published 465-row labeled CSV; not bundled.")
    parser.add_argument("--xml-dir", required=True, help="Directory containing PaDEL fingerprint XML definitions.")
    parser.add_argument("--out", default="outputs/published_study_benchmark")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _padel_features(data: pd.DataFrame, xml_file: Path) -> tuple[pd.DataFrame, list[str]]:
    from padelpy import padeldescriptor

    with tempfile.TemporaryDirectory(prefix="pfdhodh_padel_") as temp_dir:
        temp = Path(temp_dir)
        input_path = temp / "molecules.smi"
        output_path = temp / "features.csv"
        input_path.write_text(
            "".join(f"{row.canonical_smiles} {row.molecule_chembl_id}\n" for row in data.itertuples()),
            encoding="utf-8",
        )
        padeldescriptor(
            mol_dir=str(input_path),
            d_file=str(output_path),
            descriptortypes=str(xml_file),
            detectaromaticity=True,
            standardizenitro=True,
            standardizetautomers=False,
            removesalt=True,
            fingerprints=True,
            retainorder=True,
            threads=1,
            log=True,
        )
        features = pd.read_csv(output_path)
    if "Name" not in features or features["Name"].astype(str).tolist() != data["molecule_chembl_id"].astype(str).tolist():
        raise ValueError(f"PaDEL output order/IDs did not match source rows for {xml_file.name}.")
    return features.drop(columns="Name"), list(features.columns.drop("Name"))


def _morgan_features(smiles: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    matrix = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {value}")
        fp = generator.GetFingerprint(molecule)
        row = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, row)
        matrix.append(row)
    return np.asarray(matrix)


def _oversample(features: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError("Training partition must contain both classes.")
    majority = int(counts.max())
    rng = np.random.default_rng(seed)
    selected = list(range(len(labels)))
    for label, count in zip(classes, counts):
        members = np.flatnonzero(labels == label)
        if count < majority:
            selected.extend(rng.choice(members, majority - int(count), replace=True).tolist())
    order = rng.permutation(selected)
    return features[order], labels[order]


def _metric_values(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "mcc": float(matthews_corrcoef(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
    }


def main() -> None:
    args = parse_args()
    reference_path, xml_dir, out_dir = Path(args.reference_data), Path(args.xml_dir), Path(args.out)
    data = pd.read_csv(reference_path)
    required = {"molecule_chembl_id", "canonical_smiles", "class"}
    if not required.issubset(data.columns):
        raise ValueError(f"Reference CSV must contain {sorted(required)}")
    data = data.copy()
    data["activity_label"] = data["class"].str.lower().map({"inactive": 0, "active": 1})
    if data["activity_label"].isna().any() or data["canonical_smiles"].duplicated().any():
        raise ValueError("Reference labels must be binary and structures unique.")
    data["activity_label"] = data["activity_label"].astype(int)

    padel_specs = {
        "PaDEL PubChem RF": "PubchemFingerprinter.xml",
        "PaDEL SubstructureCount RF": "SubstructureFingerprintCount.xml",
    }
    feature_sets = {name: _padel_features(data, xml_dir / xml_name)[0].to_numpy(dtype=np.float32) for name, xml_name in padel_specs.items()}
    feature_sets["Morgan RF"] = _morgan_features(data["canonical_smiles"].astype(str).tolist())
    feature_hashes = {
        name: hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
        for name, matrix in feature_sets.items()
    }
    labels = data["activity_label"].to_numpy(dtype=int)
    rows, split_rows = [], []
    out_dir.mkdir(parents=True, exist_ok=True)

    for repeat in range(args.repeats):
        seed = args.seed + repeat
        splits, groups = scaffold_split_indices(data, seed)
        split_path = out_dir / "splits" / f"repeat_{repeat + 1:02d}_split_ids.csv"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        assignment = {int(index): name for name, indices in splits.items() for index in indices}
        pd.DataFrame({
            "molecule_chembl_id": data["molecule_chembl_id"].astype(str),
            "split": [assignment[index] for index in range(len(data))],
        }).to_csv(split_path, index=False)
        for name, indices in splits.items():
            split_rows.append({
                "repeat": repeat + 1,
                "seed": seed,
                "split": name,
                "molecules": len(indices),
                "active": int(labels[indices].sum()),
                "inactive": int(len(indices) - labels[indices].sum()),
                "scaffolds": int(len(set(groups[indices]))),
            })
        train, test = splits["train"], splits["test"]
        for model_name, matrix in feature_sets.items():
            selector = VarianceThreshold(threshold=0.1)
            x_train = selector.fit_transform(matrix[train])
            x_test = selector.transform(matrix[test])
            x_train, y_train = _oversample(x_train, labels[train], seed)
            model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
            model.fit(x_train, y_train)
            probs = model.predict_proba(x_test)[:, 1]
            rows.append({"repeat": repeat + 1, "seed": seed, "model": model_name, "features_retained": int(x_train.shape[1]), **_metric_values(labels[test], probs)})
        print(f"Completed repeat {repeat + 1}/{args.repeats}")

    metrics = pd.DataFrame(rows)
    splits_frame = pd.DataFrame(split_rows)
    metrics.to_csv(out_dir / "metrics_by_repeat.csv", index=False)
    splits_frame.to_csv(out_dir / "split_summary.csv", index=False)
    aggregate = metrics.groupby("model").agg({
        column: ["mean", "std"] for column in ("roc_auc", "average_precision", "mcc", "accuracy", "f1", "precision", "recall")
    })
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate.to_csv(out_dir / "summary.csv")
    plot_metrics = ("roc_auc", "mcc")
    models = metrics["model"].drop_duplicates().tolist()
    x = np.arange(len(plot_metrics))
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    for index, model_name in enumerate(models):
        group = metrics.loc[metrics["model"] == model_name]
        means = [group[metric].mean() for metric in plot_metrics]
        deviations = [group[metric].std(ddof=1) for metric in plot_metrics]
        axis.bar(x + (index - (len(models) - 1) / 2) * width, means, width, yerr=deviations, capsize=3, label=model_name)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set(xticks=x, xticklabels=["ROC-AUC", "Matthews correlation"], ylabel="Score", ylim=(-0.2, 1.05))
    axis.legend(frameon=False)
    axis.set_title("Matched scaffold-held-out benchmark")
    figure.tight_layout()
    figure.savefig(out_dir / "matched_benchmark.png", dpi=200)
    plt.close(figure)
    metadata = {
        "reference_filename": reference_path.name,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "reference_row_count": len(data),
        "class_counts": {"active": int(labels.sum()), "inactive": int(len(labels) - labels.sum())},
        "seed_start": args.seed,
        "repeats": args.repeats,
        "split": "60/20/20 Bemis-Murcko scaffold holdout; same assignments for all models",
        "feature_filter": "VarianceThreshold(0.1), fit on training partition only",
        "balancing": "Random oversampling of training partition only",
        "random_forest": {"n_estimators": 100, "probability_cutoff": 0.5},
        "padel_fingerprints": padel_specs,
        "padel_threads": 1,
        "padel_tautomer_standardization": False,
        "xml_sha256": {
            name: hashlib.sha256((xml_dir / filename).read_bytes()).hexdigest()
            for name, filename in padel_specs.items()
        },
        "feature_matrix_sha256": feature_hashes,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "padelpy": importlib.metadata.version("padelpy"),
        },
        "notes": [
            "This is a focused reanalysis, not a reproduction of the paper's 12 x 12 model search.",
            "The reference CSV and PaDEL XML files are external inputs and are not copied into this output.",
            "The published notebook uses a random 80/20 split in code; this benchmark instead uses shared scaffold-disjoint splits.",
            "PaDEL tautomer standardization is disabled because repeated runs produced differing fingerprint matrices; input canonical SMILES are retained.",
        ],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote matched benchmark to {out_dir}")


if __name__ == "__main__":
    main()
