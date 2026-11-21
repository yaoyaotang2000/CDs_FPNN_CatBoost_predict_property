# -*- coding: utf-8 -*-
"""
Model comparison using an existing carbon-dot feature table.

This script DOES NOT generate features. It directly reads the previously saved
cd_hydrothermal_features.csv/xlsx, compares regression models using repeated
3x5-fold cross-validation, and ranks them by macro-average test R2.

Default input priority
----------------------
1. ./cd_cv_outputs/cd_hydrothermal_features.csv
2. ./cd_cv_outputs/cd_hydrothermal_features.xlsx
3. ./data/cd_hydrothermal_features.csv
4. ./data/cd_hydrothermal_features.xlsx
5. ./cd_hydrothermal_features.csv
6. ./cd_hydrothermal_features.xlsx

Run
---
python compare_models_from_existing_features.py

or specify the feature file explicitly:

python compare_models_from_existing_features.py \
    --feature-file ./cd_cv_outputs/cd_hydrothermal_features.csv

Outputs
-------
./model_comparison_results/
    model_comparison_fold_metrics.csv
    model_comparison_summary.csv
    model_ranking.csv
    best_model_report.txt
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")

# ============================================================
# 1. Configuration
# ============================================================
SEED = 42
N_SPLITS = 5
N_REPEATS = 3

TARGET_COLS = ["Ex_max_nm", "Em_max_nm", "QY_%", "stoke_shift"]

# Text/meta columns that must not enter the model.
META_COLS = {
    "precursor_1",
    "precursor_2",
    "reagent_1",
    "precursor_1_key",
    "precursor_2_key",
    "reagent_1_key",
    "p1_compound_name",
    "p1_compound_key",
    "p1_SMILES",
    "p2_compound_name",
    "p2_compound_key",
    "p2_SMILES",
    "compound_name",
    "compound_key",
    "SMILES",
    "candidate_id",
    "sample_id",
    "id",
}

# Prevent target leakage from duplicated/calculated target columns.
LEAKAGE_COLS = {
    "stoke_shift_calc_check",
    "Stokes_shift_nm",
    "Stokes_shift",
    "QY_norm",
    "Stokes_norm",
    "objective",
    "score",
}

DEFAULT_FEATURE_FILES = [
    Path("./cd_cv_outputs/cd_hydrothermal_features.csv"),
    Path("./cd_cv_outputs/cd_hydrothermal_features.xlsx"),
]

DEFAULT_OUT_DIR = Path("./cd_cv_outputs")


# ============================================================
# 2. Data loading and validation
# ============================================================
def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Feature file not found: {path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature-file type: {path.suffix}")


def resolve_feature_file(user_path: str | None) -> Path:
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"Specified feature file does not exist: {path}")
        return path

    for path in DEFAULT_FEATURE_FILES:
        if path.exists():
            return path

    searched = "\n".join(f"  - {p}" for p in DEFAULT_FEATURE_FILES)
    raise FileNotFoundError(
        "No existing feature table was found. Searched:\n" + searched
    )


def prepare_xy(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, object]]:
    df = df_raw.copy()

    # Generate Stokes shift only when the feature table does not already contain it.
    # This is target preparation, not feature generation.
    if "stoke_shift" not in df.columns:
        required = {"Ex_max_nm", "Em_max_nm"}
        if not required.issubset(df.columns):
            raise ValueError(
                "stoke_shift is absent and cannot be calculated because "
                "Ex_max_nm or Em_max_nm is missing."
            )
        df["stoke_shift"] = (
            pd.to_numeric(df["Em_max_nm"], errors="coerce")
            - pd.to_numeric(df["Ex_max_nm"], errors="coerce")
        )

    missing_targets = [c for c in TARGET_COLS if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    # Remove rows with incomplete targets so every model uses identical samples.
    for c in TARGET_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=TARGET_COLS).reset_index(drop=True)
    n_after_target_drop = len(df)

    excluded = set(TARGET_COLS) | META_COLS | LEAKAGE_COLS
    candidate_cols = [c for c in df.columns if c not in excluded]

    # Convert all candidate columns to numeric. Completely non-numeric columns
    # are removed rather than silently becoming all-zero features.
    numeric_data: Dict[str, pd.Series] = {}
    dropped_non_numeric: List[str] = []
    for c in candidate_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() == 0:
            dropped_non_numeric.append(c)
            continue
        numeric_data[c] = s

    if not numeric_data:
        raise ValueError("No usable numeric feature columns were found.")

    X = pd.DataFrame(numeric_data, index=df.index)

    # Replace infinities, then median-impute using the whole table only for
    # deterministic preprocessing. Tree and linear model fitting remain inside CV.
    # For strict deployment pipelines, this imputation can be moved inside each fold.
    X = X.replace([np.inf, -np.inf], np.nan)
    medians = X.median(axis=0, numeric_only=True)
    X = X.fillna(medians).fillna(0.0)

    # Drop duplicate and constant features to avoid unnecessary computation.
    duplicate_mask = X.T.duplicated(keep="first")
    dropped_duplicate = X.columns[duplicate_mask].tolist()
    X = X.loc[:, ~duplicate_mask]

    nunique = X.nunique(dropna=False)
    dropped_constant = nunique[nunique <= 1].index.tolist()
    X = X.drop(columns=dropped_constant)

    y = df[TARGET_COLS].astype(float).copy()

    info = {
        "n_rows_raw": int(n_before),
        "n_rows_used": int(n_after_target_drop),
        "n_rows_dropped_missing_target": int(n_before - n_after_target_drop),
        "n_features_used": int(X.shape[1]),
        "dropped_non_numeric_columns": dropped_non_numeric,
        "dropped_duplicate_features": dropped_duplicate,
        "dropped_constant_features": dropped_constant,
    }
    return X, y, TARGET_COLS.copy(), info


# ============================================================
# 3. Models
# ============================================================
def build_models(seed: int, n_train: int, n_features: int) -> Dict[str, object]:
    pls_components = max(1, min(10, n_train - 1, n_features))

    models: Dict[str, object] = {
        "Ridge": Pipeline([
            ("x_scaler", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "PLS": Pipeline([
            ("x_scaler", StandardScaler()),
            (
                "model",
                PLSRegression(
                    n_components=pls_components,
                    scale=False,
                    max_iter=1000,
                ),
            ),
        ]),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500,
            max_features=0.70,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGradientBoosting": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=400,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                l2_regularization=1.0,
                random_state=seed,
            ),
            n_jobs=1,
        ),
        "CatBoost": CatBoostRegressor(
            loss_function="MultiRMSE",
            eval_metric="MultiRMSE",
            iterations=1200,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5.0,
            random_strength=1.0,
            bootstrap_type="Bernoulli",
            subsample=0.90,
            grow_policy="SymmetricTree",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = MultiOutputRegressor(
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=800,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=3,
                subsample=0.90,
                colsample_bytree=0.80,
                reg_alpha=0.05,
                reg_lambda=5.0,
                random_state=seed,
                n_jobs=1,
                verbosity=0,
            ),
            n_jobs=1,
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = MultiOutputRegressor(
            LGBMRegressor(
                objective="regression",
                n_estimators=800,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=10,
                subsample=0.90,
                colsample_bytree=0.80,
                reg_alpha=0.05,
                reg_lambda=5.0,
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
            ),
            n_jobs=1,
        )

    return models


# ============================================================
# 4. Metrics and repeated CV
# ============================================================
def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Iterable[str],
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    r2_values: List[float] = []
    normalized_mae_values: List[float] = []
    normalized_rmse_values: List[float] = []

    for j, target in enumerate(target_names):
        yt = y_true[:, j]
        yp = y_pred[:, j]

        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp))

        # Scale-independent errors for cross-target aggregation.
        scale = float(np.std(yt, ddof=0))
        scale = scale if scale > 1e-12 else 1.0
        nmae = mae / scale
        nrmse = rmse / scale

        result[f"{target}_MAE"] = mae
        result[f"{target}_RMSE"] = rmse
        result[f"{target}_R2"] = r2
        result[f"{target}_NMAE"] = nmae
        result[f"{target}_NRMSE"] = nrmse

        r2_values.append(r2)
        normalized_mae_values.append(nmae)
        normalized_rmse_values.append(nrmse)

    result["R2_macro"] = float(np.mean(r2_values))
    result["NMAE_macro"] = float(np.mean(normalized_mae_values))
    result["NRMSE_macro"] = float(np.mean(normalized_rmse_values))
    result["R2_worst_target"] = float(np.min(r2_values))
    return result


def run_model_comparison(
    X: pd.DataFrame,
    y: pd.DataFrame,
    target_names: List[str],
) -> pd.DataFrame:
    rkf = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=SEED,
    )

    rows: List[Dict[str, object]] = []

    for run_id, (train_idx, test_idx) in enumerate(rkf.split(X), start=1):
        repeat = (run_id - 1) // N_SPLITS + 1
        fold = (run_id - 1) % N_SPLITS + 1
        fold_seed = SEED + run_id

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Target scaling is fitted only on the outer training fold.
        # This prevents larger-scale wavelength targets from dominating
        # multi-output model fitting.
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train)

        models = build_models(
            seed=fold_seed,
            n_train=len(train_idx),
            n_features=X.shape[1],
        )

        for model_name, estimator in models.items():
            model = clone(estimator)
            model.fit(X_train, y_train_scaled)

            pred_scaled = np.asarray(model.predict(X_test), dtype=float)
            if pred_scaled.ndim == 1:
                pred_scaled = pred_scaled.reshape(-1, len(target_names))
            pred = y_scaler.inverse_transform(pred_scaled)

            metrics = regression_metrics(
                y_true=y_test.to_numpy(dtype=float),
                y_pred=pred,
                target_names=target_names,
            )

            record: Dict[str, object] = {
                "model": model_name,
                "repeat": repeat,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
            record.update(metrics)
            rows.append(record)

            print(
                f"[{model_name:20s}] repeat={repeat}/{N_REPEATS}, "
                f"fold={fold}/{N_SPLITS}, "
                f"R2_macro={metrics['R2_macro']:.4f}"
            )

    return pd.DataFrame(rows)


def summarize_and_rank(
    fold_df: pd.DataFrame,
    target_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        c for c in fold_df.columns
        if c not in {"model", "repeat", "fold", "n_train", "n_test"}
    ]

    summary_rows: List[Dict[str, object]] = []
    for model_name, group in fold_df.groupby("model", sort=False):
        row: Dict[str, object] = {
            "model": model_name,
            "n_cv_folds": len(group),
        }
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Primary selection: higher macro-average R2.
    # Tie breakers: lower R2 variability, higher worst-target R2,
    # and lower normalized RMSE.
    ranking_df = summary_df.sort_values(
        by=[
            "R2_macro_mean",
            "R2_macro_std",
            "R2_worst_target_mean",
            "NRMSE_macro_mean",
        ],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)

    ranking_df.insert(0, "rank", np.arange(1, len(ranking_df) + 1))
    ranking_df["selected_as_best"] = ranking_df["rank"].eq(1)

    compact_cols = [
        "rank",
        "model",
        "R2_macro_mean",
        "R2_macro_std",
        "R2_worst_target_mean",
        "NMAE_macro_mean",
        "NRMSE_macro_mean",
        "selected_as_best",
    ]
    for target in target_names:
        compact_cols.extend([
            f"{target}_R2_mean",
            f"{target}_R2_std",
            f"{target}_MAE_mean",
            f"{target}_RMSE_mean",
        ])

    ranking_df = ranking_df[compact_cols]
    return summary_df, ranking_df


# ============================================================
# 5. Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare models using an existing feature table only."
    )
    parser.add_argument(
        "--feature-file",
        type=str,
        default=None,
        help="Path to existing cd_hydrothermal_features.csv/xlsx.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directory for model-comparison outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_file = resolve_feature_file(args.feature_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"Loading existing feature table: {feature_file}")
    feature_df = read_table(feature_file)
    X, y, target_names, data_info = prepare_xy(feature_df)

    print(f"Rows used: {len(X)}")
    print(f"Features used: {X.shape[1]}")
    print(f"Targets: {target_names}")
    print(f"Models available: XGBoost={HAS_XGBOOST}, LightGBM={HAS_LIGHTGBM}")
    print("=" * 88)

    fold_df = run_model_comparison(X, y, target_names)
    summary_df, ranking_df = summarize_and_rank(fold_df, target_names)

    fold_path = out_dir / "model_comparison_fold_metrics.csv"
    summary_path = out_dir / "model_comparison_summary.csv"
    ranking_path = out_dir / "model_ranking.csv"
    report_path = out_dir / "best_model_report.txt"
    data_info_path = out_dir / "data_preparation_info.json"

    fold_df.to_csv(fold_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    data_info_path.write_text(
        json.dumps(data_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best = ranking_df.iloc[0]
    report_lines = [
        "Model comparison result",
        "=" * 72,
        f"Feature file: {feature_file}",
        f"Samples: {len(X)}",
        f"Features: {X.shape[1]}",
        f"CV: RepeatedKFold({N_SPLITS} folds x {N_REPEATS} repeats)",
        "Primary selection metric: highest mean macro-R2",
        "",
        f"Best model: {best['model']}",
        f"R2_macro: {best['R2_macro_mean']:.4f} ± {best['R2_macro_std']:.4f}",
        f"Worst-target mean R2: {best['R2_worst_target_mean']:.4f}",
        f"NMAE_macro: {best['NMAE_macro_mean']:.4f}",
        f"NRMSE_macro: {best['NRMSE_macro_mean']:.4f}",
        "",
        "Full ranking:",
        ranking_df[
            [
                "rank",
                "model",
                "R2_macro_mean",
                "R2_macro_std",
                "R2_worst_target_mean",
                "NRMSE_macro_mean",
            ]
        ].to_string(index=False),
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 88)
    print("MODEL RANKING")
    print("=" * 88)
    print(
        ranking_df[
            [
                "rank",
                "model",
                "R2_macro_mean",
                "R2_macro_std",
                "R2_worst_target_mean",
                "NRMSE_macro_mean",
            ]
        ].to_string(index=False)
    )
    print("=" * 88)
    print(f"Selected best model: {best['model']}")
    if best["model"] == "CatBoost":
        print("Conclusion: CatBoost is the best model under the stated CV protocol.")
    else:
        print(
            "Conclusion: CatBoost is not ranked first under the stated CV protocol; "
            "do not describe it as the best model without further fair tuning."
        )

    print("\nSaved files:")
    for path in [fold_path, summary_path, ranking_path, report_path, data_info_path]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
