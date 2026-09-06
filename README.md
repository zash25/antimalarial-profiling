# Antimalarial Profiling

High-throughput antimalarial compound profiling with ChEMBL bioactivity data and ChemBERTa.

## Project Overview

This project is building a reproducible Python pipeline to retrieve ChEMBL activity data on chemical molecules and organisms, prepare SMILES/activity datasets, fine-tune ChemBERTa, and evaluate antimalarial compound prediction models.

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

For the current target-discovery phase, `pip install -r requirements.txt` is enough if you already have a suitable Python environment.

## Quick Start

Identify a ChEMBL target and write selected metadata to `data/target_metadata.json`:

```bash
python antimalarial_profile.py --query "dihydroorotate dehydrogenase"
```

Useful options:

```bash
python antimalarial_profile.py --query "dihydroorotate dehydrogenase" --organism "Plasmodium falciparum"
python antimalarial_profile.py --query "lactate dehydrogenase" --organism "Plasmodium falciparum" --out data/target_metadata.json
```

Planned downstream workflow:

```bash
python scripts/extract_chembl_pf_dhodh.py --target-chembl-id CHEMBL3486 --standard-type IC50 --out data/raw/pf_dhodh_ic50_raw.csv
python scripts/preprocess_smiles.py --in data/raw/pf_dhodh_ic50_raw.csv --out data/processed/pf_dhodh_clean.csv
python train/fine_tune_chemberta.py --train data/processed/pf_dhodh_clean.csv --model outputs/chemberta_pf_dhodh
```

## Data Extraction and Preprocessing

- ChEMBL target discovery uses `chembl_webresource_client` to search by target preferred name and organism.
- Activity extraction will retrieve standardized measures such as IC50 for the selected `target_chembl_id`.
- RDKit preprocessing will canonicalize SMILES, remove invalid molecules, handle salts/fragments, deduplicate compounds, and prepare labels.
- Labels may use a configurable IC50 threshold, such as 1 uM, or regression targets from transformed activity values.

## Model Fine-Tuning

The intended model workflow uses HuggingFace/PyTorch ChemBERTa models over curated SMILES strings. Training will use reproducible train/validation/test splits, tracked random seeds, and metrics such as ROC-AUC, PR-AUC, F1, precision, and recall.

## Repository Layout

- `antimalarial_profile.py`: public CLI entry point.
- `scripts/`: ChEMBL extraction, molecule enrichment, and preprocessing utilities.
- `train/`: model training and evaluation scripts.
- `data/`: small metadata plus local raw/processed data outputs.
- `notebooks/`: optional exploratory analysis.
- `outputs/`: generated reports, figures, checkpoints, and model artifacts.
- `requirements.txt`: Python package dependencies.

## Current Status

Implemented:

- ChEMBL client dependency.
- Public target-discovery CLI.
- PfDHODH target discovery with `CHEMBL3486` selected as the current default case-study target.

Next steps:

- Implement activity extraction.
- Enrich activity rows with molecule metadata and canonical SMILES.
- Add preprocessing, labeling, baseline models, ChemBERTa fine-tuning, and final evaluation outputs.
