"""Smoke-test the ChEMBL webresource client and save visible outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from chembl_webresource_client.new_client import new_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default="user_outputs/phase_01_environment",
        help="Directory for transient user-visible outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = new_client.target
    sample = list(
        target.filter(pref_name__icontains="dihydroorotate").only(
            ["target_chembl_id", "pref_name", "organism", "target_type"]
        )[:5]
    )

    metadata = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "package": "chembl_webresource_client",
        "version": version("chembl_webresource_client"),
        "sample_query": "target.filter(pref_name__icontains='dihydroorotate')",
        "sample_count": len(sample),
        "sample_records": sample,
    }

    (out_dir / "client_check_sample.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "client_check.txt").write_text(
        "\n".join(
            [
                "ChEMBL client smoke test",
                f"Package version: {metadata['version']}",
                f"Checked at UTC: {metadata['checked_at_utc']}",
                f"Sample records returned: {metadata['sample_count']}",
                "Status: OK",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"ChEMBL client OK; wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
