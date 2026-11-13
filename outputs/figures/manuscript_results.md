# Final Model Results

## ChemBERTa
Repeated Bemis-Murcko scaffold holdout; values are mean +/- sample standard deviation.
- ROC_AUC: 0.464 +/- 0.137
- PR_AUC: 0.527 +/- 0.100
- F1: 0.487 +/- 0.169

## Morgan random forest
Repeated Bemis-Murcko scaffold holdout; values are mean +/- sample standard deviation.
- ROC_AUC: 0.648 +/- 0.117
- PR_AUC: 0.705 +/- 0.090
- F1: 0.555 +/- 0.150

## Matched reanalysis of the 2025 PfDHODH QSAR dataset
Five shared scaffold-held-out repeats on the released 465-row labeled dataset; the external reference CSV is not redistributed.
- PaDEL PubChem random forest: ROC_AUC 0.825 +/- 0.077; MCC 0.509 +/- 0.175.
- Morgan random forest: ROC_AUC 0.797 +/- 0.116; MCC 0.412 +/- 0.257.
- PaDEL SubstructureCount random forest: ROC_AUC 0.771 +/- 0.064; MCC 0.456 +/- 0.074.
- Record audit: 281 shared molecule IDs had matching structures and labels; 181 additional reference IDs were censored-only in the local raw extract; 3 were absent.

These are retrospective, scaffold-split-sensitive results, not a full reproduction of the paper's model search or prospective validation.
