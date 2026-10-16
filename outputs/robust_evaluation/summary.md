# Repeated Scaffold Evaluation

- Dataset molecules: 368
- Repeats: 5
- Split: Bemis-Murcko scaffold groups, kept disjoint across train, validation, and test.
- Reported values: mean +/- sample standard deviation across repeats.
- These repeated splits share molecules across repeats; the standard deviation describes split sensitivity and is not a confidence interval.
- Target: CHEMBL3486; activity: IC50; active threshold: 1000.0 nM.
- Labeled dataset SHA-256: `9556fe16fea95750feaffaf4aca51dc4b4af46c25dfea5cd7f8430e7d7370efd`

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Morgan random forest | 0.648 +/- 0.117 | 0.705 +/- 0.090 | 0.555 +/- 0.150 | 0.719 +/- 0.098 | 0.494 +/- 0.215 |
| ChemBERTa | 0.464 +/- 0.137 | 0.527 +/- 0.100 | 0.487 +/- 0.169 | 0.482 +/- 0.135 | 0.500 +/- 0.211 |
