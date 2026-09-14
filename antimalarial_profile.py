"""Public CLI entry point for ChEMBL-to-ChemBERTa compound profiling."""

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
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run discovery, extraction, curation, analysis, baseline, and ChemBERTa training.",
    )
    parser.add_argument("--standard-type", default="IC50", help="ChEMBL activity measure for --run-all.")
    parser.add_argument("--threshold-nm", type=float, default=1000.0, help="Active IC50 threshold for --run-all.")
    parser.add_argument("--chemberta-epochs", type=int, default=3, help="Fine-tuning epochs for --run-all.")
    parser.add_argument(
        "--skip-chemberta", action="store_true", help="Finish through the baseline model only."
    )
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
    if not args.run_all:
        return

    from scripts._utils import load_selected_target
    from scripts.enrich_chembl_molecules import main as enrich_main
    from scripts.extract_chembl_activity import main as extract_main
    from scripts.make_labels import main as label_main
    from scripts.make_manuscript_figures import main as figures_main
    from scripts.plot_dataset import main as plot_main
    from scripts.preprocess_smiles import main as preprocess_main
    from scripts.summarize_dataset import main as summarize_main
    from train.evaluate_model import main as evaluate_main
    from train.train_baseline import main as baseline_main

    target_id = load_selected_target(args.out)
    stem = f"{target_id.lower()}_{args.standard_type.lower()}"
    raw_path = f"data/raw/{stem}_raw.csv"
    enriched_path = f"data/raw/{stem}_with_smiles.csv"
    clean_path = f"data/processed/{stem}_clean.csv"
    labeled_path = f"data/processed/{stem}_labeled.csv"
    extract_main(["--metadata", args.out, "--standard-type", args.standard_type, "--out", raw_path])
    enrich_main(["--in", raw_path, "--out", enriched_path])
    preprocess_main(["--in", enriched_path, "--out", clean_path, "--rejected-out", f"data/processed/{stem}_rejected_rows.csv"])
    label_main(["--in", clean_path, "--out", labeled_path, "--threshold-nm", str(args.threshold_nm)])
    summarize_main(["--in", labeled_path, "--out", "outputs/reports/dataset_summary.json"])
    plot_main(["--in", labeled_path, "--out", "outputs/figures"])
    baseline_main(["--data", labeled_path, "--out", "outputs/baselines/random_forest", "--seed", "42"])
    evaluate_main(["--data", labeled_path, "--model", "outputs/baselines/random_forest", "--out", "outputs/reports/baseline_metrics.json"])
    if not args.skip_chemberta:
        from train.fine_tune_chemberta import main as chemberta_main

        chemberta_main(["--data", labeled_path, "--out", "outputs/chemberta_target", "--seed", "42", "--epochs", str(args.chemberta_epochs)])
        evaluate_main(["--data", labeled_path, "--model", "outputs/chemberta_target", "--out", "outputs/reports/chemberta_metrics.json"])
    figures_main(["--out", "outputs/figures"])
    print("Completed full profiling workflow. Inspect user_outputs/ for phase-by-phase results.")


if __name__ == "__main__":
    main()
