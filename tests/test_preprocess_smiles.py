from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.preprocess_smiles import canonicalize_smiles, main as preprocess_main


def test_canonicalize_smiles_keeps_largest_fragment() -> None:
    assert canonicalize_smiles("CCO.Cl") == "CCO"


def test_preprocess_normalizes_and_rejects_invalid_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    clean = tmp_path / "clean.csv"
    rejected = tmp_path / "rejected.csv"
    pd.DataFrame(
        [
            {"activity_id": "1", "assay_chembl_id": "A1", "molecule_chembl_id": "M1", "target_chembl_id": "T1", "standard_type": "IC50", "standard_value": "1", "standard_units": "uM", "standard_relation": "=", "canonical_smiles": "CCO.Cl"},
            {"activity_id": "2", "assay_chembl_id": "A1", "molecule_chembl_id": "M2", "target_chembl_id": "T1", "standard_type": "IC50", "standard_value": "0", "standard_units": "nM", "standard_relation": "=", "canonical_smiles": "CCC"},
        ]
    ).to_csv(source, index=False)
    preprocess_main(["--in", str(source), "--out", str(clean), "--rejected-out", str(rejected), "--preview-dir", str(tmp_path / "preview")])
    output = pd.read_csv(clean)
    rejected_rows = pd.read_csv(rejected)
    assert output.loc[0, "canonical_smiles"] == "CCO"
    assert output.loc[0, "standard_value_nM"] == 1000.0
    assert rejected_rows.loc[0, "rejection_reason"] == "invalid_standard_value"
