"""Shared helpers for pipeline scripts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_selected_target(metadata_path: str | Path) -> str:
    payload = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    target_id = (payload.get("selected_target") or {}).get("target_chembl_id")
    if not target_id:
        raise ValueError(
            f"No selected target in {metadata_path}. Run target discovery first."
        )
    return str(target_id)


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in records for key in row})
    if not fieldnames:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (dict, list)) else value
                    for key, value in record.items()
                }
            )


def update_observation_log(phase: str, lines: list[str]) -> None:
    """Append a compact, dated note to the private progress log."""
    log_path = Path("user_outputs/OBSERVATION_LOG.md")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry = [f"## {phase} - {utc_now()}", *[f"- {line}" for line in lines], ""]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(entry))
