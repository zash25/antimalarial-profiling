from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.make_labels import main as make_labels_main


def test_make_labels_uses_inclusive_threshold(tmp_path: Path) -> None:
    source = tmp_path / "clean.csv"
    output = tmp_path / "labeled.csv"
    pd.DataFrame(
        {"canonical_smiles": ["CCO", "CCC"], "standard_value_nM": [1000.0, 1000.1]}
    ).to_csv(source, index=False)
    make_labels_main(["--in", str(source), "--out", str(output), "--threshold-nm", "1000", "--preview-dir", str(tmp_path / "preview")])
    assert pd.read_csv(output)["activity_label"].tolist() == [1, 0]
