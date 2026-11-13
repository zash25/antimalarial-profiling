# Reference Dataset Reconciliation

The comparison used the authors' public `dhodh_05_bioactivity_data_2class_pIC50.csv` as a local external input. The file is not redistributed in this repository because the source repository has no explicit license file.

- Reference rows: 465 (202 active; 263 inactive).
- Local CHEMBL3486 primary structures shared by molecule ChEMBL ID: 281.
- Shared IDs with matching canonical SMILES: 281 of 281.
- Shared IDs with matching active/inactive labels: 281 of 281.
- Reference IDs outside the 281-structure exact dataset: 184.
- Of those 184, 181 occur in the local 602-row raw extract as right-censored-only records (`>`); three IDs do not occur in that extract.
- Reference CSV SHA-256: `142f133884b7fc142dea5c7707ceff0d513ae9fe6bf54e085f2fd51e05e61876`.

The audit reconciles molecule identity and binary labels for the shared set. It does not claim that all upstream curation decisions in the two projects were identical. The matched model metrics are in `summary.csv`; split-level scores, ID-only split assignments, feature hashes, and run settings are included alongside them.
