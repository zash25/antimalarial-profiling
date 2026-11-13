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

The complete command writes raw and curated files under `data/`, model artifacts under `outputs/`, and private phase-by-phase previews under ignored `user_outputs/`. The bundled CHEMBL3486 raw and processed files are tracked reference data; custom generated datasets remain ignored unless explicitly selected for publication.

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
- `data/`: bundled CHEMBL3486 raw and curated case-study data and target metadata.
- `outputs/`: published figures, compact evaluation summaries, per-repeat metrics, and split assignments. Model checkpoints and bulky intermediate training artifacts remain ignored.
- `outputs/published_study_benchmark/`: reproducible matched comparison results against selected fingerprints from the 2025 PfDHODH QSAR study. The third-party labeled input table is not bundled.
- `scripts/`: ChEMBL extraction, molecule enrichment, and preprocessing utilities.
- `tests/`: run tests on making labels and preprocessed smiles.
- `train/`: model training and evaluation scripts.
- `user_outputs/`: [optionally] create this folder to view artefacts generated at each phase completion.
- `requirements.txt`: Python package dependencies.

## Reference Run

The bundled PfDHODH case study selected `CHEMBL3486`. Its recorded run retrieved 602 IC50 rows, curated 368 unique molecules, and labeled 201 active versus 167 inactive compounds at 1000 nM. Five scaffold-disjoint repeats yielded mean ROC-AUC / average precision of 0.648 / 0.705 for the Morgan random forest and 0.464 / 0.527 for ChemBERTa. A focused matched reanalysis on the 2025 study's 465-row labeled set yielded scaffold-held-out mean ROC-AUC of 0.825 for PaDEL PubChem, 0.797 for Morgan, and 0.771 for PaDEL SubstructureCount. All results are retrospective and split-sensitive, from one target, and do not establish prospective model performance or experimentally validated inhibitors. ChEMBL updates can change live retrieval results.

The grouped evaluation is saved in `outputs/robust_evaluation/summary.md`; the matched benchmark table, record-reconciliation note, run metadata, and figure are in `outputs/published_study_benchmark/`. Phase-by-phase visual previews under `user_outputs/` are local-only and are not part of the public results release.

## Reproduce the Matched Benchmark

The matched reanalysis uses the authors' public 465-row dataset as an external input; it is not redistributed here because the source repository has no explicit license file. Obtain `dhodh_05_bioactivity_data_2class_pIC50.csv` from [the authors' repository](https://github.com/correctchemist/PfDHODH_QSAR_Project). Obtain PaDEL's `fingerprints_xml.zip` definitions from [PaDEL fingerprint definitions](https://github.com/dataprofessor/padel). Install the optional wrapper and a Java runtime (Java 17 or newer), then extract the XML files:

```bash
pip install -r requirements-benchmark.txt
python scripts/benchmark_published_pfdhodh.py --reference-data path/to/dhodh_05_bioactivity_data_2class_pIC50.csv --xml-dir path/to/fingerprints_xml --out outputs/published_study_benchmark
```

The script applies identical five-repeat Bemis–Murcko scaffold splits to PaDEL PubChem, PaDEL SubstructureCount, and Morgan random forests. Variance filtering and random oversampling are fitted/applied to training partitions only. PaDEL tautomer standardization is disabled because it generated different feature matrices across repeated runs; nitro/salt handling remains enabled and the reference canonical SMILES are retained. It writes aggregate and per-repeat metrics, ID-only split assignments, provenance hashes, and a figure. It is a focused common-split reanalysis, not a reproduction of the published 12-by-12 model search. The notebook's random 80/20 split code differs from the article's description of a Kennard–Stone split.

## Data and Citation

The bundled case-study data and compact analysis outputs are intentionally tracked for comparison and reproduction; review their size and provenance before adding other targets. Do not commit model checkpoints, caches, temporary descriptors, or private `user_outputs/` artifacts. Cite [ChEMBL](https://www.ebi.ac.uk/chembl/) and the ChemBERTa model used in an analysis, and record the query, target ID, activity type, threshold, split seed, and retrieval date with reported results.
