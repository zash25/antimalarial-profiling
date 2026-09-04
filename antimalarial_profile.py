"""Public CLI entry point for antimalarial compound profiling setup tasks."""

from __future__ import annotations

import argparse

from scripts.find_chembl_target import main as find_target_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find a ChEMBL antimalarial target and save metadata for later "
            "activity extraction."
        )
    )
    parser.add_argument("--query", required=True, help="Target name search term.")
    parser.add_argument(
        "--organism",
        default="Plasmodium falciparum",
        help="Organism name to match. Defaults to Plasmodium falciparum.",
    )
    parser.add_argument(
        "--out",
        default="data/target_metadata.json",
        help="Output JSON path. Defaults to data/target_metadata.json.",
    )
    parser.add_argument(
        "--preview-dir",
        default="user_outputs/phase_02_target_discovery",
        help="Optional folder for human-readable previews and candidate snapshots.",
    )
    parser.add_argument("--limit", type=int, default=50, help="Maximum targets to inspect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    find_target_main(
        [
            "--query",
            args.query,
            "--organism",
            args.organism,
            "--out",
            args.out,
            "--out-dir",
            args.preview_dir,
            "--limit",
            str(args.limit),
        ]
    )


if __name__ == "__main__":
    main()
