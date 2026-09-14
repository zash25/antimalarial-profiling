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

Run the complete workflow from discovery through final figures:

```bash
python antimalarial_profile.py --query "dihydroorotate dehydrogenase" --run-all
```

The complete command creates raw and curated data in ignored `data/raw/` and `data/processed/`, trained models under `outputs/`, and private phase-by-phase previews under ignored `user_outputs/`.

Other useful options:

```bash
python antimalarial_profile.py --query "dihydroorotate dehydrogenase" --organism "Plasmodium falciparum"
python antimalarial_profile.py --query "lactate dehydrogenase" --organism "Plasmodium falciparum" --out data/target_metadata.json
python antimalarial_profile.py --query "dihydroorotate dehydrogenase" --run-all --threshold-nm 500 --chemberta-epochs 5
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
```

## Data Extraction and Preprocessing

- ChEMBL target discovery uses `chembl_webresource_client` to search by target preferred name and organism.
- Activity extraction retrieves the requested standardized measure, defaulting to IC50, for the selected `target_chembl_id`.
- RDKit preprocessing canonicalizes SMILES, keeps the largest salt fragment, rejects non-positive/censored/unsupported measurements, converts units to nM, and retains the median IC50 per canonical SMILES.
- Binary labels use a configurable inclusive IC50 threshold; the default is 1000 nM (1 uM).

## Model Fine-Tuning

The model workflow uses HuggingFace/PyTorch ChemBERTa over curated SMILES alongside a Morgan fingerprint random-forest baseline. Both use fixed-seed stratified train/validation/test splits and report ROC-AUC, PR-AUC, F1, precision, and recall.

## Repository Layout

- `antimalarial_profile.py`: public CLI entry point.
- `.gitignore`: contains list of files and folders to be ignored during commits.
- `requirements.txt`: list of libraries and packages to install.
- `data/`: small metadata plus local raw/processed data outputs.
- `outputs/`: generated reports, figures, checkpoints, and model artifacts; ignored by Git.
- `scripts/`: ChEMBL extraction, molecule enrichment, and preprocessing utilities.
- `tests/`: run tests on making labels and preprocessed smiles.
- `train/`: model training and evaluation scripts.
- `user_outputs/`: [optionally] create this folder to view artefacts generated at each phase completion.
- `requirements.txt`: Python package dependencies.

## Reference Run

The bundled PfDHODH case study selected `CHEMBL3486`. Its recorded run retrieved 602 IC50 rows, curated 368 unique molecules, and labeled 201 active versus 167 inactive compounds at 1000 nM. Results can change as ChEMBL updates and are not a claim of general model performance.

## Data and Citation

Live data comes from ChEMBL through `chembl_webresource_client`; do not commit large retrieved datasets, model checkpoints, or private `user_outputs/` artifacts. Cite [ChEMBL](https://www.ebi.ac.uk/chembl/) and the ChemBERTa model used in your analysis or manuscript, and record the query, target ID, activity type, threshold, split seed, and retrieval date with reported results.
