"""Validate, standardize, and deduplicate ChEMBL SMILES and IC50 values."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from scripts._utils import update_observation_log
except ModuleNotFoundError:
    from _utils import update_observation_log


UNIT_TO_NM = {"PM": 0.001, "NM": 1.0, "UM": 1_000.0, "MM": 1_000_000.0, "M": 1_000_000_000.0}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, help="SMILES-enriched activity CSV.")
    parser.add_argument("--out", required=True, help="Clean, deduplicated CSV output.")
    parser.add_argument(
        "--rejected-out",
        default="data/processed/rejected_rows.csv",
        help="CSV containing excluded rows and reasons.",
    )
    parser.add_argument(
        "--preview-dir",
        default="user_outputs/phase_05_data_curation",
        help="Directory for curation notes and previews.",
    )
    return parser.parse_args(argv)


def canonicalize_smiles(smiles: str) -> str | None:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    fragments = Chem.GetMolFrags(molecule, asMols=True)
    largest_fragment = max(fragments, key=lambda fragment: fragment.GetNumHeavyAtoms())
    return Chem.MolToSmiles(largest_fragment, canonical=True, isomericSmiles=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw = pd.read_csv(args.input_path, dtype=str).fillna("")
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []

    for _, row in raw.iterrows():
        record = row.to_dict()
        value = pd.to_numeric(record.get("standard_value"), errors="coerce")
        unit = str(record.get("standard_units", "")).upper().strip()
        relation = str(record.get("standard_relation", "")).strip()
        smiles = str(record.get("canonical_smiles", "")).strip()
        reason = ""
        if pd.isna(value) or value <= 0:
            reason = "invalid_standard_value"
        elif unit not in UNIT_TO_NM:
            reason = f"unsupported_unit:{unit or 'missing'}"
        elif relation not in {"", "="}:
            reason = f"censored_relation:{relation}"
        elif not smiles:
            reason = "missing_smiles"
        else:
            canonical_smiles = canonicalize_smiles(smiles)
            if canonical_smiles is None:
                reason = "invalid_smiles"
        if reason:
            rejected.append({**record, "rejection_reason": reason})
            continue
        value_nm = float(value) * UNIT_TO_NM[unit]
        accepted.append(
            {
                **record,
                "canonical_smiles": canonical_smiles,
                "standard_value_nM": value_nm,
                "pIC50": 9 - __import__("math").log10(value_nm),
            }
        )

    if not accepted:
        raise ValueError("No valid rows remained after curation. Inspect rejected_rows.csv.")
    valid = pd.DataFrame(accepted)
    grouped = valid.groupby("canonical_smiles", as_index=False).agg(
        standard_value_nM=("standard_value_nM", "median"),
        pIC50=("pIC50", "median"),
        source_activity_count=("canonical_smiles", "size"),
        assay_count=("assay_chembl_id", "nunique"),
        molecule_chembl_id=("molecule_chembl_id", "first"),
        target_chembl_id=("target_chembl_id", "first"),
        standard_type=("standard_type", "first"),
        activity_id_example=("activity_id", "first"),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)
    rejected_path = Path(args.rejected_out)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rejected).to_csv(rejected_path, index=False)

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        "# Data Curation Summary",
        "",
        f"- Raw rows: {len(raw)}",
        f"- Accepted raw rows: {len(valid)}",
        f"- Rejected raw rows: {len(rejected)}",
        f"- Deduplicated molecules: {len(grouped)}",
        "- Duplicate policy: retain the median IC50 per canonical SMILES.",
        "- Salt policy: retain the largest RDKit-connected fragment.",
        f"- Clean dataset: `{out_path}`",
        f"- Rejected rows: `{rejected_path}`",
    ]
    (preview_dir / "curation_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    grouped.head(25).to_csv(preview_dir / "clean_data_preview.csv", index=False)
    update_observation_log(
        "Phase 5: Data Curation",
        [
            f"Curated {len(raw)} raw records into {len(grouped)} deduplicated molecules.",
            f"Rejected {len(rejected)} records; reasons are in {rejected_path}.",
            f"Inspect {preview_dir / 'curation_summary.md'} and clean_data_preview.csv.",
        ],
    )
    print(f"Curated {len(raw)} records into {len(grouped)} molecules")


if __name__ == "__main__":
    main()
