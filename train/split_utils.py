"""Shared molecule-level and scaffold-level split utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def scaffold_groups(dataset: pd.DataFrame) -> np.ndarray:
    groups = []
    for smiles in dataset["canonical_smiles"].astype(str):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES while generating scaffold split: {smiles}")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
        groups.append(scaffold or "ACYCLIC")
    return np.asarray(groups, dtype=object)


def indices_from_splits(dataset: pd.DataFrame, split_ids: str | Path) -> dict[str, np.ndarray]:
    split_frame = pd.read_csv(split_ids)
    required = {"canonical_smiles", "split"}
    if not required.issubset(split_frame.columns):
        raise ValueError(f"Split file must include columns: {sorted(required)}")
    if split_frame["canonical_smiles"].duplicated().any():
        raise ValueError("Each molecule must appear exactly once in the split file.")
    if set(split_frame["split"].astype(str)) != {"train", "validation", "test"}:
        raise ValueError("Split assignments must use train, validation, and test only.")
    if dataset["canonical_smiles"].duplicated().any():
        raise ValueError("Model input must contain one row per canonical SMILES.")

    expected = set(dataset["canonical_smiles"].astype(str))
    observed = set(split_frame["canonical_smiles"].astype(str))
    if expected != observed:
        raise ValueError("Split file molecule IDs do not exactly match the input dataset.")
    positions = {smiles: index for index, smiles in enumerate(dataset["canonical_smiles"].astype(str))}
    result = {
        name: np.asarray(
            [positions[smiles] for smiles in split_frame.loc[split_frame["split"] == name, "canonical_smiles"].astype(str)],
            dtype=int,
        )
        for name in ("train", "validation", "test")
    }
    if any(len(indices) == 0 for indices in result.values()):
        raise ValueError("Train, validation, and test splits must all contain molecules.")
    labels = dataset["activity_label"].to_numpy(dtype=int)
    if any(len(np.unique(labels[result[name]])) != 2 for name in ("train", "validation", "test")):
        raise ValueError("Every split must contain active and inactive molecules.")
    return result


def random_stratified_indices(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    train_val, test = train_test_split(
        np.arange(len(labels)), test_size=0.2, random_state=seed, stratify=labels
    )
    train, validation = train_test_split(
        train_val, test_size=0.25, random_state=seed, stratify=labels[train_val]
    )
    return {"train": train, "validation": validation, "test": test}


def _split_score(
    holdout: np.ndarray,
    total: int,
    labels: np.ndarray,
    target_fraction: float,
    global_positive_fraction: float,
) -> float:
    if not len(holdout) or len(np.unique(labels[holdout])) != 2:
        return float("inf")
    fraction_error = abs(len(holdout) / total - target_fraction)
    positive_fraction_error = abs(float(labels[holdout].mean()) - global_positive_fraction)
    return fraction_error + 0.35 * positive_fraction_error


def scaffold_split_indices(
    dataset: pd.DataFrame, seed: int, candidates: int = 256
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Choose a grouped 60/20/20 split close to target sizes and class balance."""
    labels = dataset["activity_label"].to_numpy(dtype=int)
    groups = scaffold_groups(dataset)
    if len(np.unique(groups)) < 5:
        raise ValueError("At least five distinct scaffolds are required for scaffold evaluation.")
    global_positive_fraction = float(labels.mean())

    outer = GroupShuffleSplit(n_splits=candidates, test_size=0.2, random_state=seed)
    best_outer: tuple[np.ndarray, np.ndarray] | None = None
    best_outer_score = float("inf")
    for train_val, test in outer.split(np.zeros(len(dataset)), labels, groups):
        score = _split_score(test, len(dataset), labels, 0.2, global_positive_fraction)
        if len(np.unique(labels[train_val])) != 2:
            score = float("inf")
        if score < best_outer_score:
            best_outer, best_outer_score = (train_val, test), score
    if best_outer is None or not np.isfinite(best_outer_score):
        raise ValueError("Could not create a scaffold test split containing both activity classes.")

    train_val, test = best_outer
    inner = GroupShuffleSplit(n_splits=candidates, test_size=0.25, random_state=seed + 1)
    best_inner: tuple[np.ndarray, np.ndarray] | None = None
    best_inner_score = float("inf")
    local_groups = groups[train_val]
    local_labels = labels[train_val]
    for train_local, validation_local in inner.split(
        np.zeros(len(train_val)), local_labels, local_groups
    ):
        train, validation = train_val[train_local], train_val[validation_local]
        score = _split_score(validation, len(dataset), labels, 0.2, global_positive_fraction)
        if len(np.unique(labels[train])) != 2:
            score = float("inf")
        if score < best_inner_score:
            best_inner, best_inner_score = (train, validation), score
    if best_inner is None or not np.isfinite(best_inner_score):
        raise ValueError("Could not create a scaffold validation split containing both classes.")

    train, validation = best_inner
    splits = {"train": train, "validation": validation, "test": test}
    scaffold_sets = {name: set(groups[indices]) for name, indices in splits.items()}
    if any(scaffold_sets[left] & scaffold_sets[right] for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise RuntimeError("Scaffold leakage detected between split partitions.")
    return splits, groups


def write_split_file(dataset: pd.DataFrame, indices: dict[str, np.ndarray], path: str | Path) -> None:
    assignments = {
        int(index): name
        for name, selected in indices.items()
        for index in selected
    }
    frame = pd.DataFrame(
        {
            "canonical_smiles": dataset["canonical_smiles"].astype(str),
            "split": [assignments[index] for index in range(len(dataset))],
        }
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
