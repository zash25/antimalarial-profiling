# Antimalarial Profiling: A Plain-Language Guide

## The Project in Brief

This project collects previously published measurements about compounds tested against a malaria parasite enzyme, organizes the chemical data, and checks whether computer models can recognize compounds that inhibit that enzyme. It is a research and prioritization tool, not a newly discovered or approved malaria treatment.

## What the Project Studies

The example analysis focuses on *Plasmodium falciparum* dihydroorotate dehydrogenase (PfDHODH), a parasite enzyme. Its ChEMBL identifier is `CHEMBL3486`. ChEMBL is a public database of chemical and biological research results.

The code requested 602 published IC50 measurements for this target and processed them into 368 unique molecules. Some records could not be used or represented duplicate measurements of the same molecule. IC50 is the amount of a compound needed to reduce the measured enzyme activity by half in a particular laboratory test. A lower IC50 generally means stronger inhibition in that test. It does not, by itself, tell us whether a compound treats malaria in a person or even kills the whole parasite.

At a chosen cutoff of 1,000 nanomolar (1 micromolar), the dataset labels 201 molecules as active and 167 as less active. This cutoff is a practical way to create two groups for this analysis; it is not a universal definition of a useful drug.

## What the Code Does

The command-line program lets a user choose a biological target and run the available analysis. In broad strokes, the code:

1. Retrieves reported activity measurements and chemical structures from ChEMBL.
2. Checks and standardizes structures, removes unusable records, and assigns active or less-active labels using the selected cutoff.
3. Trains two models on part of the labeled data.
4. Tests their predictions on held-out molecules and saves tables, summaries, and model outputs.

The saved results for the repeated evaluation are in [`user_outputs/phase_11_robust_evaluation/summary.md`](user_outputs/phase_11_robust_evaluation/summary.md). The manuscript-ready metric summary is in [`outputs/figures/manuscript_results.md`](outputs/figures/manuscript_results.md).

## The Models and Their Training

The **Morgan random forest** converts each molecule into a compact description of its local chemical features, then combines the decisions of many decision trees to estimate whether it belongs to the active group. It was trained on 2,048-bit molecular fingerprints using 500 trees.

**ChemBERTa** is a model first trained to recognize patterns in text-like representations of molecules. This project further trained, or "fine-tuned," it for three passes over the activity data so it could predict the active and less-active labels.

Training means showing a model examples with known labels so it can learn patterns. It does not mean the model has learned a proven rule of biology. The project also includes small automated software checks for chemical-structure processing and label assignment. Those checks verify parts of the code; they are not laboratory experiments.

## How the Models Were Tested

The evaluation was repeated five times. In each repeat, the test set was kept separate from training, and related molecules with the same basic chemical framework were kept together rather than split between the two. This is a more demanding test of whether a model can handle unfamiliar chemical families. It is different from testing the model in a laboratory.

The table shows average scores across those repeats. Higher is generally better. ROC-AUC measures how well a model ranks active molecules above less-active ones; around 0.5 is roughly random ranking. The plus/minus values show how much scores varied across the five data splits, not a formal confidence interval.

| Model | ROC-AUC | PR-AUC | F1 score |
| --- | ---: | ---: | ---: |
| Morgan random forest | 0.648 +/- 0.117 | 0.705 +/- 0.090 | 0.555 +/- 0.150 |
| ChemBERTa | 0.464 +/- 0.137 | 0.527 +/- 0.100 | 0.487 +/- 0.169 |

In this dataset and evaluation, the Morgan random forest was the more promising of the two. Its results were moderate and varied between repeats, so it should be treated as a tool for early prioritization, not as a dependable decision-maker. ChemBERTa did not show reliable ability to rank unfamiliar molecules in this test.

## Molecules and Specific Findings

The lowest IC50 records in the retrieved dataset were for these ChEMBL molecules:

| ChEMBL molecule | Recorded IC50 against PfDHODH |
| --- | ---: |
| [CHEMBL2431496](https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBL2431496/) | 6 nM |
| [CHEMBL4545127](https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBL4545127/) | 10 nM |
| [CHEMBL2012961](https://www.ebi.ac.uk/chembl/compound_report_card/CHEMBL2012961/) | 17 nM |

These are already-known compounds with reported enzyme-assay measurements in ChEMBL. They are the strongest measurements observed in this particular retrieved dataset, not molecules newly discovered or tested by this project. Their assay methods, repeatability, selectivity, effects on whole parasites, safety, and suitability as medicines would all need further investigation.

## How This Can Help Drug Discovery

Laboratory teams can test only a limited number of compounds at a time. A workflow like this can organize existing evidence and help rank which compounds or chemical families may be worth examining next. Its scaffold-based test also gives an early warning if a model only recognizes structures similar to examples it has already seen.

The project does **not** show that any listed molecule cures malaria, is safe, reaches the parasite in the body, or is better than existing medicines. Those questions require carefully designed laboratory studies and, much later, clinical research. The useful result here is a reproducible analysis of public data, a comparison of two prediction approaches, and a starting point for better validated follow-up work.
