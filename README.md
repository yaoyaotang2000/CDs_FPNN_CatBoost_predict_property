# CDs_FPNN_CatBoost_predict_property

Code and numerical data for *Ensemble Learning-Driven Closed-Loop Optimization for Accelerated Discovery of Carbon Dots*. The workflow ranks synthesis conditions for blue, green, yellow and orange emission. It predicts the excitation maximum, emission maximum and quantum yield (QY), and derives the Stokes shift from the two wavelengths.

## Workflow at a glance

1. Standardize the initial experimental data and the prospectively observed history.
2. Define synthesis success and train a five-seed CatBoost success-classifier ensemble.
3. Retain experimentally successful rows for optical-property regression.
4. Train the FPNN and target-specific CatBoost regressors.
5. Form the property ensemble:

   \[
   \hat y_{\mathrm{ens}}=0.5\hat y_{\mathrm{FPNN}}+0.5\hat y_{\mathrm{CatBoost}}.
   \]

6. Search the predefined mixed discrete synthesis space using success-constrained BO–UCB.
7. Experimentally evaluate selected recipes, append the observations, retrain the models, and repeat.
8. Evaluate the prospective loop using prediction errors, hit rates, Pareto-front expansion, and hypervolume gain.
## Repository organization

```text
.
├── closed_loop.ipynb                  Jupyter entry point
├── run_cycle.py                      prediction and complete-cycle commands
├── scripts/
│   ├── 01_update_training_data_from_history.py
│   ├── 02_train_fp_catboost_cycle.py
│   ├── 03_ax_bo_search_cycle.py
│   ├── success1_repeated_cv_generated_features.py
│   ├── weight_sensitivity_success_filtered_3runs.py
│   ├── analyze_history_pareto_hv.py
│   ├── explain_fpnn_catboost_ensemble.py
│   ├── fp_process_encoder_ablation_learning_curve.py
│   └── fp_encoder_dimension_ablation_ensemble.py
├── data/                             initial experiments, structures and validation endpoints
├── cd_catboost_feature_selection/    ordered feature lists for the three properties
├── success_feature_selection_independent/  classifier feature list
├── model/                            model.py and package initializer
├── workflow/                         handoff between the three numbered stages
├── checkpoints/                      cycle-1 and cycle-17 fitted ensembles and scalers
├── cd_cv_outputs/                    descriptor workbook used in the validation analyses
├── feature_inputs/                   initial-data feature matrix
├── validation/                       independent comparisons and their original model variants
├── analysis/                         calculations added for peer review
├── results/                          archived numerical evidence for the paper
├── outputs/                          generated when running the documented entry points
├── requirements.txt
└── UPLOAD_MANIFEST.csv                paths, byte sizes and SHA-256 checksums
```

The numbered entry points in `scripts/` contain the update, fit and search code extracted from the experimental notebook. The similarly named `02` and `03` files in `validation/` are the versions used in separate validation studies. Their callers explicitly load those versions.

## Data

| File | Contents and use |
| --- | --- |
| [Initial experiments](data/output_with_molar_amount_pubchem.xlsx) | 224 labeled recipes with synthesis conditions and optical measurements. |
| [Compound structures](data/compound_smiles.xlsx) | Chemical names and SMILES used to construct descriptors and fingerprints. |
| [Prospective endpoints](data/observed_history.csv) | Predictions and measurements for 33 in-protocol recommendations, and ensemble dispersion for the 24 rows with a verified match. Used for errors, strict hits, stopping trajectories and hypervolume. |

The release includes the initial recipes, the two trained ensembles and the numerical tables for the reported results. The laboratory's round-by-round recipe history and production logs remain local. The 33 endpoint rows support outcome analysis. Continuing the closed loop requires new measured recipes with the input fields described below.

### Success definition

The recorded `success` column supplies the training labels. In the initial workbook all 145 positive labels have QY ≥ 4%, and all 79 negative labels have QY < 4%. The regressors use the successful subset. The classifier trains on both classes with ten selected features. Its mean probability across five fitted models is an uncalibrated optical success score.

## Property surrogate model

The FPNN encodes each precursor with 167 MACCS keys and 512 Avalon bits. Each 679-element fingerprint passes through a shared 128-to-32 encoder. The model combines the sum and absolute difference of the two latent vectors with an embedding of 14 synthesis features and returns three optical properties. The CatBoost models use recipe, solvent, structural and process descriptors, including radius-2, 256-bit Morgan fingerprints. The ordered feature lists contain 20 features for excitation, 35 for emission and 35 for QY.

Each branch uses five fitted members in the full model. The final property estimate is the arithmetic mean of the FPNN and CatBoost branch means. The archived checkpoints include the target scaler and the feature order for the fitted models.

## Model evaluation

```console
python scripts/success1_repeated_cv_generated_features.py
```

This command compares FPNN, CatBoost and three FPNN/CatBoost mixtures (30/70, 50/50 and 70/30) on the 145 successful initial recipes, with the same 3 × 5 random cross-validation folds for all models. Outputs include held-out predictions, fold-level metrics and summaries under `outputs/random_cv/`. Selected features and architecture were informed by development data. Table S21 tests the fixed model under separate random and precursor-pair holdouts. Its script and archived pooled metrics are listed below.

## Success-constrained BO–UCB

For each color the code normalizes QY and Stokes shift to fixed bounds of 0–100% and 40–180 nm. A Gaussian penalty with a width of 10 nm reduces the score outside the target window. The acquisition score multiplies the success probability by the sum of the expected score and β times a dispersion term derived from the ensemble. The default QY weight α is 0.70 and β is 0.20. Candidates with a success score below 0.50 have zero acquisition value.

The initial table contains 28 precursor identities, including polyaniline. The notebook's Ax search lists 27 identities and omits polyaniline. The two precursor positions each accept five masses (0.2–1.0 g in 0.2 g steps). There are five solvents at 15 mL, four temperatures (160–220 °C in 20 °C steps) and three durations (4, 6 or 8 h). This yields **28 × 28 × 5 × 5 × 5 × 4 × 3 = 1,176,000 ordered parameter combinations** before equivalence and previously tested recipes are considered. Identical precursor identities are allowed. Tested recipes are matched using unordered precursor–mass pairs.

The final configuration calls Ax managed optimization with 200 trials per run, five runs per active color, base seed 42 and one recommendation per active color. Ax selects its generation strategy. Historical settings varied across fits. The acquisition interval for blue is 430–480 nm. The strict experimental hit definition in Table S8 uses 440–480 nm for blue, 500–540 for green, 560–580 for yellow and 580–630 for orange, inclusive for the intended target.

## Closed-loop update and stopping

Start with **[closed_loop.ipynb](closed_loop.ipynb)** after installing the dependencies below. The same three computational stages can be run individually from `scripts/`.

For the initial cycle, execute these commands in order. Steps 02 and 03 automatically select the latest prepared run. Supply the printed directory using `--run-dir` to choose a specific run.

```console
python scripts/01_update_training_data_from_history.py --cycle 1
python scripts/02_train_fp_catboost_cycle.py --cycle 1
python scripts/03_ax_bo_search_cycle.py --cycle 1
```

Stopping is evaluated per target color once a measured emission has reached its search window. The final settings use a ceiling of 16 cycles and a patience of four cycles after at least three completed cycles. An improvement must exceed the greater of 0.02 score units and 3% of the previous best. The paper's retrospective stopping tables apply the evaluation windows to the measured endpoints. The additional seventh yellow measurement lies outside the 33-row cohort.

## Sensitivity and robustness analyses



### Paper figures and additional review analyses


The new review calculations write to `outputs/revision/`. The numerical results supplied with the manuscript are in `results/revision/`. Some original development scripts in `validation/` write to their own output directories or regenerate feature lists. Use a separate checkout when repeating feature selection.

## Installation

Use Python 3.10. The verification environment used Python 3.10.16 on Windows. Run these commands from the repository root.

```console
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux, then run:

```console
python -m pip install -r requirements.txt
python run_cycle.py predict
```

Open the notebook with `python -m jupyterlab closed_loop.ipynb`. The saved-checkpoint example and endpoint statistics provide quick numerical checks. Full refits and independent searches take longer and can vary across hardware and software builds.

## Reproducibility notes

Run every entry point from the repository root. Keep input row identifiers and selected feature order intact. The checksums in `UPLOAD_MANIFEST.csv` cover every release file except the manifest itself. The public endpoint table supports evaluation of the published recommendations. Replaying their exact chronology requires the original recipe-level laboratory records.

