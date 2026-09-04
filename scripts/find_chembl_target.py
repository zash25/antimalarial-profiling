"""Find candidate ChEMBL targets for a query and organism."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TARGET_FIELDS = [
    "target_chembl_id",
    "pref_name",
    "organism",
    "target_type",
    "tax_id",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True,
                        help="Target name search term.")
    parser.add_argument(
        "--organism",
        default="Plasmodium falciparum",
        help="Organism name to match. Defaults to Plasmodium falciparum.",
    )
    parser.add_argument(
        "--out",
        default="data/target_metadata.json",
        help="Path for selected metadata JSON.",
    )
    parser.add_argument(
        "--out-dir",
        default="user_outputs/phase_02_target_discovery",
        help="Directory for transient user-visible outputs.",
    )
    parser.add_argument("--limit", type=int, default=50,
                        help="Maximum records to keep.")
    return parser.parse_args(argv)


def normalize(value: Any) -> str:
    return str(value or "").casefold()


def score_target(record: dict[str, Any], query: str, organism: str) -> int:
    pref_name = normalize(record.get("pref_name"))
    record_organism = normalize(record.get("organism"))
    query_terms = [term for term in normalize(query).split() if len(term) > 2]

    score = 0
    if normalize(organism) in record_organism:
        score += 100
    score += sum(5 for term in query_terms if term in pref_name)
    if record.get("target_type") == "SINGLE PROTEIN":
        score += 10
    return score


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2,
                    sort_keys=True), encoding="utf-8")


def find_target(args: argparse.Namespace) -> dict[str, Any]:
    from chembl_webresource_client.new_client import new_client

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = new_client.target
    broad = list(target.filter(pref_name__icontains=args.query).only(
        TARGET_FIELDS)[: args.limit])
    organism_matches = [
        record
        for record in broad
        if normalize(args.organism) in normalize(record.get("organism"))
    ]

    if not broad:
        broad = list(target.search(args.query).only(
            TARGET_FIELDS)[: args.limit])
        organism_matches = [
            record
            for record in broad
            if normalize(args.organism) in normalize(record.get("organism"))
        ]

    ranked = sorted(
        organism_matches or broad,
        key=lambda record: score_target(record, args.query, args.organism),
        reverse=True,
    )
    selected = ranked[0] if ranked else None

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "query": args.query,
        "organism": args.organism,
        "selected_target": selected,
        "candidate_count": len(broad),
        "organism_match_count": len(organism_matches),
        "candidates": broad,
    }

    write_json(out_dir / "dhodh_targets_preview.json", broad)
    write_json(out_dir / "pf_dhodh_targets_preview.json", organism_matches)
    write_json(Path(args.out), payload)

    note_lines = [
        "# Selected ChEMBL Target",
        "",
        f"- Query: `{args.query}`",
        f"- Organism: `{args.organism}`",
        f"- Candidate records: {len(broad)}",
        f"- Organism matches: {len(organism_matches)}",
    ]
    if selected:
        note_lines.extend(
            [
                f"- Selected target: `{selected.get('target_chembl_id')}`",
                f"- Preferred name: {selected.get('pref_name')}",
                f"- ChEMBL organism: {selected.get('organism')}",
                f"- Target type: {selected.get('target_type')}",
            ]
        )
    else:
        note_lines.append("- Selected target: none found")

    (out_dir / "selected_target.md").write_text("\n".join(note_lines) +
                                                "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    payload = find_target(args)
    out_dir = Path(args.out_dir)
    print(f"Wrote target discovery outputs to {out_dir}")
    print(f"Wrote selected target metadata to {args.out}")
    selected = payload.get("selected_target")
    if selected:
        print(
            "Selected target: "
            f"{selected.get('target_chembl_id')} - {selected.get('pref_name')} "
            f"({selected.get('organism')})"
        )


if __name__ == "__main__":
    main()
