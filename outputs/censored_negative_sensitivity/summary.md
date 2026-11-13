# Right-Censored Negative Sensitivity Check

- Exact-measurement molecules: 368 (201 active; 167 inactive).
- Added censored-only molecules: 182; each has a reported lower bound above 1000 nM and was labeled inactive.
- Augmented molecules: 550; cutoff: 1000 nM.
- Scaffold-disjoint repeats: 5; seeds: 42 through 46.
- Model: 500-tree Morgan radius-2, 2,048-bit random forest with balanced class weights.
- This is a secondary curation sensitivity analysis, not external validation. The one censored structure also present in the exact dataset was not added or used to override its exact-measurement label.
- Augmented dataset SHA-256: `b9963d4347c9d6bf04044806a6018c6da5d19d92544a14c1db03803dc58a69dc`

| Metric | Mean +/- sample SD |
| --- | ---: |
| ROC_AUC | 0.814 +/- 0.059 |
| PR_AUC | 0.714 +/- 0.081 |
| F1 | 0.522 +/- 0.117 |
| PRECISION | 0.759 +/- 0.203 |
| RECALL | 0.454 +/- 0.186 |
