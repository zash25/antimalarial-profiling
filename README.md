# Antimalarial Profiling

Reproducible compound profiling from ChEMBL bioactivity data to baseline and ChemBERTa models.

## Project Overview

The pipeline retrieves ChEMBL activity data, enriches and curates SMILES, creates active/inactive labels, trains a Morgan-fingerprint random forest and ChemBERTa classifier, and writes evaluation figures and metrics.

The default case study is *Plasmodium falciparum* dihydroorotate dehydrogenase (PfDHODH) inhibition using IC50 data. The CLI is intentionally configurable: users can provide another target query and organism when they want to profile a different ChEMBL target.

Example:

```bash
python antimalarial_profile.py --query "dihydroorotate dehydrogenase"
python antimalarial_profile.py --query "dihydroorotate dehydrogenase" --organism "Plasmodium falciparum"
```

## Setup

RDKit is usually easiest to install with Conda, especially on Windows.

```bash
conda create -n chemberta python=3.10 -y
conda activate chemberta
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

`requirements.txt` also supports a pure-pip setup where wheels are available. PyTorch will use CUDA automatically when a compatible GPU is installed; otherwise ChemBERTa training runs on CPU.

## Quick Start

Discover a target only:

```bash
python antimalarial_profile.py --query "dihydroorotate dehydrogenase"
```

Run the complete workflow from discovery through repeated scaffold evaluation and final figures. The default final phase trains both models over five scaffold-disjoint repeats, so it takes longer than a single training run:

```bash
python antimalarial_profile.py --query "dihydroorotate dehydrogenase" --run-all
```

The complete command creates raw and curated data in ignored `data/raw/` and `data/processed/`, trained models under `outputs/`, and private phase-by-phase previews under ignored `user_outputs/`.

Other useful options:

```bash
python antimalarial_profile.py --query "dihydroorotate dehydrogenase" --organism "Plasmodium falciparum"
python antimalarial_profile.py --query "lactate dehydrogenase" --organism "Plasmodium falciparum" --out data/target_metadata.json
python antimalarial_profile.py --query "dihydroorotate dehydrogenase" --run-all --threshold-nm 500 --chemberta-epochs 5 --evaluation-repeats 5
python antimalarial_profile.py --query "dihydroorotate dehydrogenase" --run-all --skip-chemberta
```

## Individual Commands

```bash
python scripts/extract_chembl_activity.py --target-chembl-id CHEMBL3486 --standard-type IC50 --out data/raw/chembl3486_ic50_raw.csv
python scripts/enrich_chembl_molecules.py --in data/raw/chembl3486_ic50_raw.csv --out data/raw/chembl3486_ic50_with_smiles.csv
python scripts/preprocess_smiles.py --in data/raw/chembl3486_ic50_with_smiles.csv --out data/processed/chembl3486_ic50_clean.csv
python scripts/make_labels.py --in data/processed/chembl3486_ic50_clean.csv --threshold-nm 1000 --out data/processed/chembl3486_ic50_labeled.csv
python train/train_baseline.py --data data/processed/chembl3486_ic50_labeled.csv --out outputs/baselines/random_forest
python train/fine_tune_chemberta.py --data data/processed/chembl3486_ic50_labeled.csv --out outputs/chemberta_target --seed 42
python train/evaluate_scaffold.py --data data/processed/chembl3486_ic50_labeled.csv --out outputs/robust_evaluation --repeats 5 --seed 42
```

## Data Extraction and Preprocessing

- ChEMBL target discovery uses `chembl_webresource_client` to search by target preferred name and organism.
- Activity extraction retrieves the requested standardized measure, defaulting to IC50, for the selected `target_chembl_id`.
- RDKit preprocessing canonicalizes SMILES, keeps the largest salt fragment, rejects non-positive/censored/unsupported measurements, converts units to nM, and retains the median IC50 per canonical SMILES.
- Binary labels use a configurable inclusive IC50 threshold; the default is 1000 nM (1 uM).

## Model Fine-Tuning

The model workflow uses HuggingFace/PyTorch ChemBERTa over curated SMILES alongside a Morgan fingerprint random-forest baseline. Initial diagnostics use a fixed-seed stratified split. The final evaluation repeats Bemis-Murcko scaffold-grouped 60/20/20 holdouts; both models receive the same molecule IDs for each repeat, and a scaffold is never divided between train, validation, and test. Results include per-repeat metrics, split composition, and mean plus sample standard deviation. Repeated splits overlap with one another, so that standard deviation describes split sensitivity and is not a confidence interval.

## Repository Layout

- `antimalarial_profile.py`: public CLI entry point.
- `.gitignore`: contains list of files and folders to be ignored during commits.
- `requirements.txt`: list of libraries and packages to install.
- `data/`: small metadata plus local raw/processed data outputs.
- `outputs/`: generated reports, figures, checkpoints, and model artifacts; ignored by Git.
- `outputs/robust_evaluation/`: repeated scaffold split IDs, per-repeat predictions/metrics, and aggregate results; ignored by Git.
- `scripts/`: ChEMBL extraction, molecule enrichment, and preprocessing utilities.
- `tests/`: run tests on making labels and preprocessed smiles.
- `train/`: model training and evaluation scripts.
- `user_outputs/`: [optionally] create this folder to view artefacts generated at each phase completion.
- `requirements.txt`: Python package dependencies.

## Reference Run

The bundled PfDHODH case study selected `CHEMBL3486`. Its recorded run retrieved 602 IC50 rows, curated 368 unique molecules, and labeled 201 active versus 167 inactive compounds at 1000 nM. Five scaffold-disjoint repeats yielded mean ROC-AUC / PR-AUC of 0.648 / 0.705 for the Morgan random forest and 0.464 / 0.527 for ChemBERTa. These are exploratory results from one ChEMBL target and a small dataset, not evidence of general model performance. ChEMBL updates can also change retrieved records.

The final grouped evaluation is saved in `outputs/robust_evaluation/summary.md`; the final comparison figure and manuscript summary are `outputs/figures/model_comparison.png` and `outputs/figures/manuscript_results.md`. The same summaries and a split-by-split preview are copied to `user_outputs/phase_11_robust_evaluation/` for local review.

## Data and Citation

Live data comes from ChEMBL through `chembl_webresource_client`; do not commit large retrieved datasets, model checkpoints, or private `user_outputs/` artifacts. Cite [ChEMBL](https://www.ebi.ac.uk/chembl/) and the ChemBERTa model used in your analysis or manuscript, and record the query, target ID, activity type, threshold, split seed, and retrieval date with reported results.
