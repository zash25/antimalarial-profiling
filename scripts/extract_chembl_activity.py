"""Export standardized ChEMBL activity records for a selected target."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._utils import (
        load_selected_target,
        update_observation_log,
        utc_now,
        write_csv,
    )
except ModuleNotFoundError:
    from _utils import load_selected_target, update_observation_log, utc_now, write_csv


ACTIVITY_FIELDS = [
    "activity_id",
    "assay_chembl_id",
    "assay_type",
    "bao_format",
    "canonical_smiles",
    "document_chembl_id",
    "molecule_chembl_id",
    "pchembl_value",
    "standard_relation",
    "standard_type",
    "standard_units",
    "standard_value",
    "target_chembl_id",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-chembl-id", help="ChEMBL target ID. Overrides metadata.")
    parser.add_argument(
        "--metadata", default="data/target_metadata.json", help="Target metadata JSON path."
    )
    parser.add_argument("--standard-type", default="IC50", help="Activity measure to retrieve.")
    parser.add_argument("--out", help="Raw CSV output path.")
    parser.add_argument("--limit", type=int, help="Optional cap for a quick exploratory run.")
    parser.add_argument(
        "--preview-dir",
        default="user_outputs/phase_03_activity_extraction",
        help="Directory for human-readable activity snapshots.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    target_id = args.target_chembl_id or load_selected_target(args.metadata)
    standard_type = args.standard_type.upper()
    out_path = Path(args.out or f"data/raw/{target_id.lower()}_{standard_type.lower()}_raw.csv")
    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    from chembl_webresource_client.new_client import new_client

    activities = new_client.activity.filter(
        target_chembl_id=target_id, standard_type=standard_type
    ).only(ACTIVITY_FIELDS)
    total_count = len(activities)
    if args.limit:
        activities = activities[: args.limit]
    records = [dict(record, retrieved_at_utc=utc_now()) for record in activities]

    write_csv(out_path, records)
    write_csv(preview_dir / "raw_ic50_sample.csv", records[:25])
    (preview_dir / "activity_count.txt").write_text(
        "\n".join(
            [
                f"Target: {target_id}",
                f"Standard type: {standard_type}",
                f"Total matching ChEMBL records: {total_count}",
                f"Records exported: {len(records)}",
                f"Raw CSV: {out_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    update_observation_log(
        "Phase 3: Activity Extraction",
        [
            f"Retrieved {len(records)} of {total_count} {standard_type} activity records for {target_id}.",
            f"Inspect {preview_dir / 'raw_ic50_sample.csv'} for the first 25 raw records.",
            f"Full raw table: {out_path}.",
        ],
    )
    print(f"Retrieved {len(records)} of {total_count} {standard_type} records for {target_id}")
    print(f"Wrote raw activity data to {out_path}")


if __name__ == "__main__":
    main()
