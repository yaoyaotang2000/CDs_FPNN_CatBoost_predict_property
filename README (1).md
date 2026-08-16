# Success-Constrained FPNN–CatBoost and BO–UCB for Closed-Loop Carbon-Dot Discovery

This repository contains the data-processing, surrogate-model training, Bayesian-optimization, active-learning, sensitivity-analysis, and evaluation code associated with our closed-loop discovery workflow for carbon dots (CDs). The workflow predicts excitation wavelength (`Ex_max_nm`), emission wavelength (`Em_max_nm`), quantum yield (`QY_%`), and Stokes shift, and recommends synthesis conditions for blue-, green-, yellow-, and orange-emissive CDs.

The released implementation replaces the earlier high-dimensional multimodal model with a compact fingerprint–process neural network (FPNN) and target-specific CatBoost regressors. Their predictions are combined using an equal-weight ensemble. A separate success classifier provides a feasibility constraint for Bayesian optimization (BO), while an upper-confidence-bound (UCB) term rewards informative uncertainty.

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

The public release should use the following structure. If the scripts are currently stored under different names, rename or move them before publication while preserving their contents.

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── output_with_molar_amount_pubchem.xlsx   # initial experiments
│   ├── observed_history.xlsx                   # prospective AL observations
│   └── compound_smiles.xlsx                    # precursor/solvent structures
├── cd_catboost_feature_selection/
│   ├── Ex_max_nm_best_top20_feature_list.csv
│   ├── Em_max_nm_best_top35_feature_list.csv
│   └── QY_%_best_top35_feature_list.csv
├── model/
│   ├── __init__.py
│   └── model.py
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
└── outputs/                                  # generated locally; not required as input
```

## Data

### Initial and prospective data

- `data/output_with_molar_amount_pubchem.xlsx` contains the initialization experiments.
- `data/observed_history.xlsx` contains the recipes proposed and experimentally evaluated during active learning, including cycle number, target color, predicted properties, measured properties, and success labels.
- `data/compound_smiles.xlsx` maps chemical names to structures used by the featurizer.

All row identifiers must remain stable. When an external feature table is used, rows must be joined by the explicit row identifier rather than by positional indexing.

### Success definition

An experiment is labelled unsuccessful (`success = 0`) when both of the following criteria are met:

- measured Stokes shift is below 80 nm; and
- measured quantum yield is below 4%.

All remaining labelled experiments are assigned `success = 1`. Optical-property regressors are trained only on `success = 1` rows. The success classifier is trained on both classes and is used only to assess feasibility during BO.

The current success-classification dataset contains 224 labelled experiments (145 successful and 79 unsuccessful). Candidate features are ranked using repeated cross-validation, and the selected classifier uses the top 10 features. Five final CatBoost classifiers are trained with seeds `42`, `2024`, `3407`, `777`, and `1024`. Their mean predicted probability is reported as \(P_{\mathrm{success}}\). Candidates with \(P_{\mathrm{success}}<0.50\) are rejected during prospective search.

## Property surrogate model

### FPNN component

Each precursor is represented by a 679-dimensional fingerprint. A shared neural encoder maps each fingerprint through `679 → 128 → 32`. The two latent precursor vectors, \(f_1\) and \(f_2\), are combined symmetrically as

\[
[f_1+f_2,\ |f_1-f_2|],
\]

which makes the pair representation invariant to precursor order. The resulting 64-dimensional vector is processed by a `64 → 64 → 64` fusion network. Fourteen process features (solvent descriptors, temperature, reaction time, and precursor ratio) are mapped directly to 16 dimensions. The concatenated 80-dimensional representation is passed through a `80 → 64 → 3` regression head.

Dropout is applied to the fingerprint encoder and regression head. Target values are standardized using statistics calculated from each training fold only.

### CatBoost component

CatBoost features are generated directly from the standardized recipes and molecular structures. Target-specific feature lists are used:

- excitation wavelength: top 20 selected features;
- emission wavelength: top 35 selected features;
- quantum yield: top 35 selected features.

Target columns and derived target leakage variables are excluded from the feature matrix.

### Ensemble

For every property, the final prediction is the arithmetic mean of the independently trained FPNN and CatBoost predictions. The 0.5/0.5 weight was selected from cross-validated comparisons with the individual models and the 0.3/0.7 and 0.7/0.3 alternatives.

## Model evaluation

The principal comparison uses three repeats of five-fold cross-validation (`RepeatedKFold`, 15 held-out folds) on `success = 1` data. The same fold indices are used for all models. The following models are evaluated:

- FPNN;
- CatBoost;
- 0.3 FPNN + 0.7 CatBoost;
- 0.5 FPNN + 0.5 CatBoost;
- 0.7 FPNN + 0.3 CatBoost.

For `Ex_max_nm`, `Em_max_nm`, and `QY_%`, the scripts report fold-level and aggregate MAE, RMSE, and \(R^2\), including mean ± sample standard deviation. Because all preprocessing and model fitting occur within each training fold, held-out observations do not influence feature scaling or fitting.

Run the comparison from the repository root:

```bash
python scripts/success1_repeated_cv_generated_features.py
```

## Success-constrained BO–UCB

For a candidate recipe \(x\), quantum yield and Stokes shift are normalized to fixed physically meaningful ranges. The expected color-specific score is

\[
\mu_S(x)=C_{\mathrm{color}}(x)
\left[\alpha Q(x)+(1-\alpha)\Delta(x)\right],
\]

where \(C_{\mathrm{color}}\) measures compatibility with the target emission window, \(Q\) is normalized QY, and \(\Delta\) is normalized Stokes shift. Weight sensitivity supported \(\alpha=0.7\).

Exploration is introduced through

\[
\mathrm{UCB}(x)=\mu_S(x)+\beta\sigma_S(x),
\]

where \(\sigma_S\) is estimated from prediction dispersion among independently saved ensemble members. Exploration-coefficient sensitivity supported \(\beta=0.2\). The acquisition value is

\[
A(x)=P_{\mathrm{success}}(x)
\left[\mu_S(x)+\beta\sigma_S(x)\right].
\]

The positive uncertainty term rewards, rather than penalizes, informative uncertainty. The separate success constraint prevents this reward from directing experiments toward candidates predicted to be non-emissive or otherwise unsuccessful.

### Search domain

The production script searches the finite domain defined by:

- two precursor identities, with identical precursor pairs optionally allowed;
- precursor masses: 0.2, 0.4, 0.6, 0.8, or 1.0 g;
- solvent: water, ethanol, dimethylformamide, sulfuric acid, or phosphoric acid;
- temperature: 160, 180, 200, or 220 °C;
- reaction time: 4, 6, or 8 h;
- fixed solvent volume: 15 mL.

Recipes already present in the expanded training data or observed history are excluded using an order-invariant recipe key. Duplicate recommendations are also removed before experimental selection.

### Production Ax settings

The supplied production script calls `ax.service.managed_loop.optimize`, allowing Ax to use its managed default generation strategy for the installed Ax version. The script does **not** explicitly fix the number of Sobol initialization trials, the BoTorch acquisition optimizer, `raw_samples`, or `num_restarts`; therefore, these quantities must not be reported as user-specified settings. The release environment pins the Ax version in `requirements.txt` so the default behavior is reproducible.

The explicit production settings are:

- 200 in-silico Ax trials per run;
- 5 independent Ax runs per active target color and cycle;
- 1 experimentally selected recipe per active color and cycle;
- seed 42;
- objective weight \(\alpha=0.7\);
- UCB coefficient \(\beta=0.2\);
- hard feasibility threshold \(P_{\mathrm{success}}\ge 0.50\).

The 100-trial budget used in selected sensitivity experiments is a separate diagnostic setting and is not the production BO budget.

Run one prospective search cycle after training:

```bash
python scripts/03_ax_bo_search_cycle.py
```

## Closed-loop update and stopping

After experimental evaluation, append measured results to `data/observed_history.xlsx`, increment the cycle number in the configuration, and execute:

```bash
python scripts/01_update_training_data_from_history.py
python scripts/02_train_fp_catboost_cycle.py
python scripts/03_ax_bo_search_cycle.py
```

Stopping is evaluated independently for each target color. Search continues until an experimentally measured sample enters the target emission window. Thereafter, stopping occurs when either the maximum budget of 16 cycles is reached or no improvement larger than the noise-aware tolerance is observed for four consecutive cycles after at least three cycles:

\[
\tau_t=\max\left(0.02,\ 0.03S^*_{t-1},\ 0.02\right).
\]

The two 0.02 terms represent the absolute score tolerance and the score-noise floor, respectively. All evaluated experiments, including exploratory trials that do not improve the incumbent score, remain in the observed history and are used in subsequent retraining.

## Sensitivity and robustness analyses

### Objective-weight and exploration sensitivity

```bash
python scripts/weight_sensitivity_success_filtered_3runs.py
```

The sensitivity workflow filters property-regression data to `success = 1`, uses 100 BO trials per condition, repeats each condition with three independent seeds, and reports mean ± SD. These runs support the choices \(\alpha=0.7\) and \(\beta=0.2\); they do not replace the production search budget described above.

### Model interpretation

```bash
python scripts/explain_fpnn_catboost_ensemble.py
```

The interpretation workflow reports grouped permutation importance for the complete ensemble, native CatBoost SHAP contributions, component performance, and comparisons with simple molecular-proxy models. These analyses describe predictive associations and should not be interpreted as causal mechanistic evidence.

### Architecture and learning-curve analyses

```bash
python scripts/fp_process_encoder_ablation_learning_curve.py
python scripts/fp_encoder_dimension_ablation_ensemble.py
```

These scripts compare process-encoder depth/width, fingerprint bottleneck dimensions, precursor-fusion strategies, and performance as a function of training-set size using fixed held-out folds.

### Pareto-front and hypervolume evaluation

```bash
python scripts/analyze_history_pareto_hv.py \
  --initial data/output_with_molar_amount_pubchem.xlsx \
  --history data/observed_history.xlsx
```

This analysis compares the experimentally observed initialization and cumulative post-optimization Pareto fronts using measured QY and Stokes shift. It reports pooled and color-resolved Pareto counts, normalized hypervolume, bootstrap confidence intervals, absolute gain, and relative gain.

## Installation

Python 3.10 is recommended. Create a clean environment and install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU execution, install the appropriate PyTorch build for the local CUDA version before installing the remaining packages. The released experiments can also be run on CPU, although repeated cross-validation and sensitivity analyses will be slower.

## Reproducibility notes

- Run all commands from the repository root so relative paths resolve correctly.
- Do not modify row identifiers when editing spreadsheets.
- Record the full configuration and software environment for every cycle.
- Random seeds are set for Python, NumPy, PyTorch, and model training where supported.
- Generated run directories contain configuration manifests, selected feature lists, model files, fold-level metrics, predictions, and figures.
- Large generated models and intermediate candidate tables may be released through an archival repository if they exceed GitHub file-size limits.

## Data and code availability

The initial dataset, prospective active-learning history, molecular-structure mapping, selected feature lists, analysis scripts, and plotting code required to reproduce the reported results are provided in this repository. Trained model checkpoints and complete machine-readable outputs should be deposited in the GitHub release or a permanent archive and linked here before publication.

Repository: `https://github.com/<OWNER>/<REPOSITORY>`  
Archived release/DOI: `<TO BE ADDED>`

## Citation

If you use this workflow, please cite the associated article:

```bibtex
@article{<citation_key>,
  title   = {<article title>},
  author  = {<authors>},
  journal = {<journal>},
  year    = {<year>},
  doi     = {<doi>}
}
```

Replace all angle-bracket placeholders before making the repository public.
