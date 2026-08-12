# Antimalarial-profiling
High-throughput profiling for antimalarial compound screening using ChemBERTa data.

**Project Overview**
- **Goal:**: Provide an open-source, reproducible Python pipeline to fine-tune ChemBERTa on bioactivity data (SMILES → IC50) and rapidly screen for Plasmodium falciparum Dihydroorotate Dehydrogenase (PfDHODH) inhibitors.
- **Approach:**: Extract IC50 data from ChEMBL, preprocess SMILES with RDKit, fine-tune a ChemBERTa transformer (HuggingFace/PyTorch), and evaluate classification/regression performance.

**Manuscript Structure**
- **Abstract:**: Background, methodology (SQL extraction, SMILES preprocessing, fine-tuning), results, conclusion.
- **Introduction:**: Public-health context, ChEMBL as data source, rationale for 1D SMILES-based transformers, research objective.
- **Methodology:**: SQL & ChEMBL extraction, RDKit curation, ChemBERTa model and tokenizer, fine-tuning protocol and hyperparameters.
- **Results & Evaluation:**: ROC-AUC, precision/recall/F1, ROC/confusion plots, inference speed comparison to docking.
- **Discussion & Conclusion:**: Implications, limitations, open science statement.

**Setup & Dependencies**
- **Recommended Python environment:**: Use a virtual environment or Conda. RDKit is easiest to install via Conda.

Example (conda):
```bash
conda create -n chemberta python=3.10 -y
conda activate chemberta
conda install -c conda-forge rdkit -y
pip install -r requirements.txt
```

- **Typical dependencies:**: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `transformers`, `datasets`, `rdkit` (or `rdkit-pypi` where applicable).

**Quick Start**
- Place your local ChEMBL SQLite export (or query results) into a `data/` folder.
- Run the SQL extraction script to pull PfDHODH assays and standard IC50 values.

Example (high-level):
```bash
python scripts/extract_chembl_pf_dhodh.py --db path/to/chembl.db --out data/pf_dhodh.csv
python scripts/preprocess_smiles.py --in data/pf_dhodh.csv --out data/pf_dhodh_clean.csv
python train/fine_tune_chemberta.py --train data/pf_dhodh_clean.csv --model outputs/chemberta_pf_dhodh
```

**Data Extraction & Preprocessing**
- **SQL extraction:**: Filter ChEMBL for PfDHODH target assays and standard IC50 measurements; export SMILES and activity columns.
- **RDKit curation:**: Canonicalize SMILES, remove salts/fragments, validate molecules, and optionally standardize tautomeric forms.
- **Labeling:**: Binarize IC50 values into active/inactive using a configurable threshold (e.g., 1 µM) or treat as regression with log-transformed IC50.

**Model Fine-Tuning (High-level)**
- **Model:**: ChemBERTa (RoBERTa-like) tokenizer for SMILES and HuggingFace `transformers` fine-tuning with PyTorch.
- **Training:**: Use stratified Train/Val/Test splits, monitor ROC-AUC and F1, save best checkpoints, and log hyperparameters.

**Repository Layout (recommended)**
- `data/`: raw and processed data outputs.
- `scripts/`: SQL extraction and preprocessing scripts.
- `train/`: training and evaluation scripts for fine-tuning ChemBERTa.
- `notebooks/`: exploratory analysis, visualization, and attention mapping notebooks.
- `models/` or `outputs/`: trained checkpoints and example inference scripts.
- `requirements.txt`: pinned Python dependencies.

**Usage Notes & Next Steps**
**VS Code Workflow (Using VS Code Exclusively)**
- **Develop & run scripts:** Use the VS Code Python extension and the integrated terminal to run `scripts/` and `train/` scripts; prefer `python scripts/...` for reproducibility.
- **Environment:** Create virtualenv or Conda env from VS Code's Python selector; install `rdkit` via Conda if needed.
- **Interactive work:** Use VS Code's Interactive Window or the built-in Jupyter support for ad-hoc exploration and inline plots — no separate Jupyter server required.
- **Debugging & tasks:** Use the VS Code debugger (`launch.json`) for breakpoints and the `Tasks` system to run repeatable commands.
- **Docstrings & code:** Include triple-quoted docstrings in functions/classes — VS Code shows them on hover and they work with documentation tools.
- **Plots & visualization:** `plt.show()` opens a native window; for reproducible or headless runs, `plt.savefig()` is recommended and files can be opened from VS Code or the OS. Use the Plot Viewer for quick inspection.
- **Notebooks optional:** Notebooks are optional — prefer scripts for training/production and use Interactive Window for exploration and figures.

**Usage Notes & Next Steps**
- This README reflects the manuscript outline and intended project structure; scripts referenced above are placeholders to implement.
- Next tasks: add `requirements.txt`, add extraction and training scripts, and include example notebooks demonstrating results and plots.
