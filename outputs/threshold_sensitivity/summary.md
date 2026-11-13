# Activity-Cutoff Sensitivity Check

- Dataset molecules: 368
- Scaffold split repeats: 5; saved 1,000 nM split assignments reused at every cutoff.
- Model: 500-tree Morgan radius-2, 2,048-bit random forest with balanced class weights.
- Metrics are mean +/- sample standard deviation across repeated splits; this is split variability, not a confidence interval.
- Clean dataset SHA-256: `d6837577954e83baff013bec898911767dcba7f05129bef99212da9c3042c143`
- This is a cutoff sensitivity check for the stronger primary model, not an independent validation dataset.

| Active cutoff (nM) | Active / inactive | ROC-AUC | PR-AUC | F1 |
| ---: | ---: | ---: | ---: | ---: |
| 500 | 164 / 204 | 0.703 +/- 0.074 | 0.615 +/- 0.060 | 0.485 +/- 0.171 |
| 1000 | 201 / 167 | 0.648 +/- 0.117 | 0.705 +/- 0.090 | 0.555 +/- 0.150 |
| 2000 | 228 / 140 | 0.732 +/- 0.077 | 0.817 +/- 0.049 | 0.647 +/- 0.127 |
