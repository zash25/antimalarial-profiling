"""Add canonical SMILES and representative molecule metadata to activity records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts._utils import update_observation_log, write_csv, write_json
except ModuleNotFoundError:
    from _utils import update_observation_log, write_csv, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, help="Raw activity CSV.")
    parser.add_argument("--out", required=True, help="SMILES-enriched CSV output.")
    parser.add_argument(
        "--preview-dir",
        default="user_outputs/phase_04_molecule_metadata",
        help="Directory for user-visible molecule previews.",
    )
    return parser.parse_args(argv)


def molecule_details(molecule_id: str) -> dict[str, Any]:
    from chembl_webresource_client.new_client import new_client

    return dict(new_client.molecule.get(molecule_id))


def extract_smiles(record: dict[str, Any]) -> str:
    structures = record.get("molecule_structures") or {}
    if isinstance(structures, str):
        try:
            structures = json.loads(structures)
        except json.JSONDecodeError:
            structures = {}
    return str(structures.get("canonical_smiles") or "")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    with Path(args.input_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No activity rows found in {args.input_path}.")

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, dict[str, Any]] = {}
    enriched: list[dict[str, Any]] = []
    missing_smiles = 0

    for row in rows:
        molecule_id = row.get("molecule_chembl_id", "")
        smiles = row.get("canonical_smiles", "")
        details: dict[str, Any] = {}
        if molecule_id and not smiles:
            details = cache.setdefault(molecule_id, molecule_details(molecule_id))
            smiles = extract_smiles(details)
        if not smiles:
            missing_smiles += 1
        enriched.append(
            {
                **row,
                "canonical_smiles": smiles,
                "molecule_pref_name": details.get("pref_name", ""),
                "molecule_max_phase": details.get("max_phase", ""),
            }
        )

    representative_id = next((row.get("molecule_chembl_id") for row in rows if row.get("molecule_chembl_id")), None)
    if representative_id:
        representative = cache.get(representative_id) or molecule_details(representative_id)
        write_json(preview_dir / "example_molecule.json", representative)
        representative_smiles = extract_smiles(representative) or next(
            (row["canonical_smiles"] for row in enriched if row.get("canonical_smiles")), ""
        )
        if representative_smiles:
            from rdkit import Chem
            from rdkit.Chem import Draw

            molecule = Chem.MolFromSmiles(representative_smiles)
            if molecule is not None:
                structure_dir = preview_dir / "molecule_structures"
                structure_dir.mkdir(parents=True, exist_ok=True)
                image = Draw.MolToImage(molecule, size=(600, 400))
                image.save(structure_dir / f"{representative_id}.png")

    write_csv(args.out, enriched)
    write_csv(preview_dir / "molecule_preview.csv", enriched[:25])
    update_observation_log(
        "Phase 4: Molecule Metadata",
        [
            f"Enriched {len(enriched)} activity records with canonical SMILES where available.",
            f"Rows still missing SMILES: {missing_smiles}.",
            f"Inspect {preview_dir / 'example_molecule.json'}, molecule_preview.csv, and molecule_structures/.",
        ],
    )
    print(f"Enriched {len(enriched)} rows; {missing_smiles} rows have no canonical SMILES")
    print(f"Wrote SMILES-enriched data to {args.out}")


if __name__ == "__main__":
    main()
