#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two training policies for the 0.5 FPNN + 0.5 CatBoost ensemble:

    1) all_rows:     use all property-measured rows for model training;
    2) success_only: remove success=0 rows from each OUTER-TRAINING fold only.

The outer test folds are identical for both policies. Three evaluation subsets
are reported:

    success_1 : primary endpoint for the success-constrained BO workflow;
    all       : supplementary generalization result;
    success_0 : diagnostic extrapolation result.

This is deliberately different from filtering the complete table before CV,
which would produce different test sets and an unfair comparison.

Expected project files
----------------------
./02_train_fp_catboost_cycle_success_selected_features.py
./data/output_with_molar_amount_pubchem.xlsx
./data/observed_history.xlsx
./data/compound_smiles.xlsx
./cd_catboost_feature_selection/*_feature_list.csv

Run
---
python compare_success_filter_fpnn_catboost.py

Outputs
-------
success_filter_comparison/
  dataset_used.csv
  split_manifest.csv
  fold_metrics.csv
  metric_summary.csv
  paired_policy_differences.csv
  out_of_fold_predictions.csv
  Fig_success_filter_comparison.png
  Fig_success_filter_comparison.pdf
  run_config.json

Notes
-----
* The script dynamically imports the production 02 script so that the exact
  FPNN architecture, fingerprint implementation, CatBoost feature generation,
  and selected feature lists are reused.
* Fixed feature lists are appropriate for isolating the effect of success-row
  filtering. If those lists were selected using this same dataset, do not call
  the resulting CV estimates fully nested feature-selection estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import importlib.util
import json
import random
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

TARGETS = ["Ex_max_nm", "Em_max_nm", "QY_%"]
POLICIES = ("all_rows", "success_only")


@dataclass
class Cfg:
    # Production model source. Change only if your 02 filename differs.
    production_02_script: str = "./validation/02_train_fp_catboost_cycle_success_selected_features.py"

    # If this exists, it is used directly. Otherwise initial + history are merged.
    expanded_training_file: str = ""
    initial_training_file: str = "./data/output_with_molar_amount_pubchem.xlsx"
    observed_history_file: str = "./data/observed_history.xlsx"
    smiles_path: str = "./data/compound_smiles.xlsx"

    output_dir: str = "./success_filter_comparison"

    # Repeated outer CV: 3 repetitions x 5 folds.
    n_repeats: int = 3
    n_splits: int = 5
    split_seeds: tuple = (42, 2024, 3407)

    # Use the five production seeds for the final manuscript calculation.
    # Reduce to (42,) only for a quick code check.
    model_seeds: tuple = (42, 2024, 3407, 777, 1024)

    nn_epochs: int = 300
    nn_lr: float = 2e-3
    nn_weight_decay: float = 1e-4
    nn_batch_size: int = 64
    nn_grad_clip: float = 1.0

    cb_iterations: int = 2000
    cb_learning_rate: float = 0.05
    cb_depth: int = 4
    cb_l2_leaf_reg: float = 5.0
    cb_random_strength: float = 1.0
    cb_subsample: float = 0.9

    success_threshold: float = 0.5
    success_stokes_cutoff_nm: float = 80.0
    success_qy_cutoff_pct: float = 4.0
    ensemble_fpnn_weight: float = 0.5
    ensemble_catboost_weight: float = 0.5
    random_seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def import_production_module(path: str):
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Production 02 script not found: {p}\n"
            "Set Cfg.production_02_script to the actual path."
        )
    spec = importlib.util.spec_from_file_location("production_02", p)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = [
        "data_process", "CarbonDotFeaturizer", "FPProcessDataset",
        "FPProcessNN", "collate_fn", "get_solvent_polarity_data",
        "build_compound_feature_table", "build_dataset_features",
        "read_feature_list", "FEATURE_LIST_FILES",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Production 02 script is missing: {missing}")
    return module


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aliases = {
        "reagent": "reagent_1",
        "mass_1": "mass_1_g",
        "mass_2": "mass_2_g",
        "Mass1(g)": "mass_1_g",
        "Mass2(g)": "mass_2_g",
        "stokes_shift": "Stokes_shift_nm",
        "stoke_shift": "Stokes_shift_nm",
        "Psuccess": "success",
        "P_success": "success",
    }
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    if "Stokes_shift_nm" not in out.columns and {
        "Em_max_nm", "Ex_max_nm"
    }.issubset(out.columns):
        out["Stokes_shift_nm"] = (
            pd.to_numeric(out["Em_max_nm"], errors="coerce")
            - pd.to_numeric(out["Ex_max_nm"], errors="coerce")
        )
    # data_process() in the production code expects this spelling.
    if "Stokes_shift_nm" in out.columns:
        out["stoke_shift"] = out["Stokes_shift_nm"]

    numeric_cols = [
        "temperature", "time_h", "pre_1_mmol", "pre_2_mmol",
        "Ex_max_nm", "Em_max_nm", "QY_%", "Stokes_shift_nm", "success",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def fill_success_label(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    """Fill only missing success labels using the manuscript definition."""
    out = df.copy()
    if "success" not in out.columns:
        out["success"] = np.nan
    available = out[["Stokes_shift_nm", "QY_%"]].notna().all(axis=1)
    missing = out["success"].isna() & available
    failure = (
        (out["Stokes_shift_nm"] < cfg.success_stokes_cutoff_nm)
        & (out["QY_%"] < cfg.success_qy_cutoff_pct)
    )
    out.loc[missing, "success"] = np.where(failure[missing], 0, 1)
    return out


def recipe_key(df: pd.DataFrame) -> pd.Series:
    """Group identical recipes to prevent recipe leakage across outer folds."""
    cols = [
        "precursor_1", "precursor_2", "reagent_1", "temperature", "time_h",
        "pre_1_mmol", "pre_2_mmol",
    ]
    keys = []
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row.get(col, np.nan)
            if col in {"temperature", "time_h", "pre_1_mmol", "pre_2_mmol"}:
                vals.append("NA" if pd.isna(val) else f"{float(val):.6f}")
            else:
                vals.append(str(val).strip().lower())
        keys.append("||".join(vals))
    return pd.Series(keys, index=df.index, name="recipe_key")


def load_modeling_data(cfg: Cfg, prod) -> pd.DataFrame:
    if cfg.expanded_training_file:
        raw = read_table(cfg.expanded_training_file)
        raw["data_source"] = raw.get("data_source", "expanded_training")
    else:
        initial = standardize_columns(read_table(cfg.initial_training_file))
        initial["data_source"] = "initial_training"
        history_path = Path(cfg.observed_history_file)
        if history_path.is_file():
            history = standardize_columns(read_table(history_path))
            history = history.dropna(subset=TARGETS).copy()
            history["data_source"] = "observed_history"
            raw = pd.concat([initial, history], ignore_index=True, sort=False)
        else:
            raw = initial

    raw = fill_success_label(standardize_columns(raw), cfg)
    # Reuse production property-data cleaning.
    cleaned = prod.data_process(raw.copy()).reset_index(drop=True)
    cleaned = fill_success_label(standardize_columns(cleaned), cfg)

    required = [
        "precursor_1", "precursor_2", "reagent_1", "temperature", "time_h",
        "pre_1_mmol", "pre_2_mmol", "success", *TARGETS,
    ]
    missing = [c for c in required if c not in cleaned.columns]
    if missing:
        raise ValueError(f"Modeling data are missing columns: {missing}")
    cleaned = cleaned.dropna(subset=required).copy()
    cleaned["success"] = (cleaned["success"] >= cfg.success_threshold).astype(int)
    cleaned["recipe_key"] = recipe_key(cleaned)
    cleaned = cleaned.drop_duplicates("recipe_key", keep="last").reset_index(drop=True)
    cleaned["row_id"] = np.arange(len(cleaned), dtype=int)
    return cleaned


def build_catboost_table(df: pd.DataFrame, cfg: Cfg, prod) -> pd.DataFrame:
    smiles_compound = pd.read_excel(cfg.smiles_path, sheet_name="compound")
    compound_features, fp_objects = prod.build_compound_feature_table(smiles_compound)
    feature_df = prod.build_dataset_features(
        df.drop(columns=["recipe_key"], errors="ignore"),
        compound_features,
        fp_objects,
        has_targets=True,
    )
    if len(feature_df) != len(df):
        raise RuntimeError(
            "CatBoost feature generation changed row count. Check failed SMILES "
            "and the production build_dataset_features function."
        )
    feature_df = feature_df.reset_index(drop=True)
    feature_df["row_id"] = df["row_id"].to_numpy()
    return feature_df


def build_outer_splits(df: pd.DataFrame, cfg: Cfg):
    y = df["success"].astype(int).to_numpy()
    groups = df["recipe_key"].astype(str).to_numpy()
    X_dummy = np.zeros((len(df), 1), dtype=float)
    splits = []
    for repeat, seed in enumerate(cfg.split_seeds[: cfg.n_repeats], start=1):
        splitter = StratifiedGroupKFold(
            n_splits=cfg.n_splits, shuffle=True, random_state=int(seed)
        )
        for fold, (tr, te) in enumerate(splitter.split(X_dummy, y, groups), start=1):
            splits.append((repeat, fold, int(seed), tr, te))
    return splits


def train_fpnn_one_seed(prod, train_df, y_scaler, cfg: Cfg, seed: int, device):
    set_seed(seed)
    smiles = pd.read_excel(cfg.smiles_path)
    featurizer = prod.CarbonDotFeaturizer(smiles, prod.get_solvent_polarity_data())
    dataset = prod.FPProcessDataset(train_df, featurizer, has_y=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.nn_batch_size,
        shuffle=True,
        collate_fn=prod.collate_fn,
        num_workers=0,
    )
    model = prod.FPProcessNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.nn_lr, weight_decay=cfg.nn_weight_decay
    )
    for _ in range(cfg.nn_epochs):
        model.train()
        for batch in loader:
            y_scaled = y_scaler.transform(
                batch["y"].numpy().astype(np.float32)
            ).astype(np.float32)
            pred = model(
                batch["fps_1"].to(device),
                batch["fps_2"].to(device),
                batch["global_feat"].to(device),
            )
            target = torch.as_tensor(y_scaled, dtype=torch.float32, device=device)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.nn_grad_clip)
            optimizer.step()
    return model, featurizer


@torch.no_grad()
def predict_fpnn(prod, model, featurizer, test_df, y_scaler, cfg: Cfg, device):
    dataset = prod.FPProcessDataset(test_df, featurizer, has_y=False)
    loader = DataLoader(
        dataset,
        batch_size=cfg.nn_batch_size,
        shuffle=False,
        collate_fn=prod.collate_fn,
        num_workers=0,
    )
    chunks = []
    model.eval()
    for batch in loader:
        scaled = model(
            batch["fps_1"].to(device),
            batch["fps_2"].to(device),
            batch["global_feat"].to(device),
        ).cpu().numpy()
        chunks.append(y_scaler.inverse_transform(scaled))
    return np.vstack(chunks)


def catboost_matrix(feature_df, features):
    X = feature_df[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X


def train_predict_policy(
    prod, cfg: Cfg, policy: str, df: pd.DataFrame, feature_df: pd.DataFrame,
    train_idx: np.ndarray, test_idx: np.ndarray, split_seed: int,
):
    outer_train = df.iloc[train_idx].copy()
    outer_test = df.iloc[test_idx].copy()
    if policy == "success_only":
        model_train = outer_train.loc[outer_train["success"] == 1].copy()
    elif policy == "all_rows":
        model_train = outer_train.copy()
    else:
        raise ValueError(policy)
    if len(model_train) < 20:
        raise RuntimeError(f"Too few training rows for {policy}: {len(model_train)}")

    y_scaler = StandardScaler().fit(model_train[TARGETS].astype(float).to_numpy())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fpnn_preds = []
    cb_preds_by_seed = []
    train_ids = model_train["row_id"].to_numpy()
    test_ids = outer_test["row_id"].to_numpy()
    f_train = feature_df.set_index("row_id").loc[train_ids].reset_index()
    f_test = feature_df.set_index("row_id").loc[test_ids].reset_index()

    selected_by_target = {}
    for target in TARGETS:
        selected_by_target[target] = prod.read_feature_list(
            prod.FEATURE_LIST_FILES[target]
        )
        missing = [f for f in selected_by_target[target] if f not in feature_df.columns]
        if missing:
            raise ValueError(f"Missing CatBoost features for {target}: {missing[:20]}")

    for model_i, base_seed in enumerate(cfg.model_seeds):
        seed = int(base_seed + 10000 * split_seed + 100 * model_i)
        model, featurizer = train_fpnn_one_seed(
            prod, model_train, y_scaler, cfg, seed, device
        )
        fpnn_preds.append(
            predict_fpnn(prod, model, featurizer, outer_test, y_scaler, cfg, device)
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cb_target_predictions = []
        for target in TARGETS:
            features = selected_by_target[target]
            model_cb = CatBoostRegressor(
                loss_function="RMSE",
                iterations=cfg.cb_iterations,
                learning_rate=cfg.cb_learning_rate,
                depth=cfg.cb_depth,
                l2_leaf_reg=cfg.cb_l2_leaf_reg,
                random_strength=cfg.cb_random_strength,
                bootstrap_type="Bernoulli",
                subsample=cfg.cb_subsample,
                grow_policy="SymmetricTree",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )
            model_cb.fit(
                catboost_matrix(f_train, features),
                f_train[target].astype(float),
                verbose=False,
            )
            cb_target_predictions.append(
                model_cb.predict(catboost_matrix(f_test, features))
            )
        cb_preds_by_seed.append(np.column_stack(cb_target_predictions))

    pred_fpnn = np.mean(np.stack(fpnn_preds), axis=0)
    pred_cb = np.mean(np.stack(cb_preds_by_seed), axis=0)
    pred_ensemble = (
        cfg.ensemble_fpnn_weight * pred_fpnn
        + cfg.ensemble_catboost_weight * pred_cb
    )
    return outer_train, model_train, outer_test, pred_fpnn, pred_cb, pred_ensemble


def metric_values(y_true, y_pred):
    if len(y_true) == 0:
        return {"r2": np.nan, "mae": np.nan, "rmse": np.nan}
    return {
        "r2": r2_score(y_true, y_pred) if len(y_true) >= 2 else np.nan,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
    }


def collect_results(
    repeat, fold, split_seed, policy, outer_train, model_train, outer_test,
    pred_fpnn, pred_cb, pred_ensemble,
):
    metric_rows, pred_rows = [], []
    subsets = {
        "all": np.ones(len(outer_test), dtype=bool),
        "success_1": outer_test["success"].to_numpy() == 1,
        "success_0": outer_test["success"].to_numpy() == 0,
    }
    components = {
        "FPNN": pred_fpnn,
        "CatBoost": pred_cb,
        "Ensemble_0.5_0.5": pred_ensemble,
    }
    for i, (_, row) in enumerate(outer_test.iterrows()):
        for j, target in enumerate(TARGETS):
            pred_rows.append({
                "repeat": repeat, "fold": fold, "split_seed": split_seed,
                "policy": policy, "row_id": int(row["row_id"]),
                "success": int(row["success"]), "target": target,
                "y_true": float(row[target]),
                "pred_fpnn": float(pred_fpnn[i, j]),
                "pred_catboost": float(pred_cb[i, j]),
                "pred_ensemble": float(pred_ensemble[i, j]),
            })
    for subset_name, mask in subsets.items():
        for component, predictions in components.items():
            for j, target in enumerate(TARGETS):
                vals = metric_values(
                    outer_test.loc[mask, target].astype(float).to_numpy(),
                    predictions[mask, j],
                )
                metric_rows.append({
                    "repeat": repeat, "fold": fold, "split_seed": split_seed,
                    "policy": policy, "evaluation_subset": subset_name,
                    "component": component, "target": target,
                    "n_outer_train": len(outer_train),
                    "n_model_train": len(model_train),
                    "n_test": int(mask.sum()), **vals,
                })
    return metric_rows, pred_rows


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["policy", "evaluation_subset", "component", "target"]
    rows = []
    for key, g in fold_metrics.groupby(id_cols, dropna=False):
        row = dict(zip(id_cols, key))
        row["n_outer_folds"] = len(g)
        row["n_model_train_mean"] = g["n_model_train"].mean()
        row["n_test_mean"] = g["n_test"].mean()
        for metric in ["r2", "mae", "rmse"]:
            row[f"{metric}_mean"] = g[metric].mean()
            row[f"{metric}_std"] = g[metric].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_differences(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    ens = fold_metrics.loc[
        fold_metrics["component"] == "Ensemble_0.5_0.5"
    ].copy()
    index_cols = ["repeat", "fold", "evaluation_subset", "target"]
    rows = []
    for metric in ["r2", "mae", "rmse"]:
        wide = ens.pivot_table(index=index_cols, columns="policy", values=metric)
        if not set(POLICIES).issubset(wide.columns):
            continue
        wide = wide.dropna(subset=list(POLICIES))
        # Positive delta always means success_only is better.
        delta = (
            wide["success_only"] - wide["all_rows"]
            if metric == "r2"
            else wide["all_rows"] - wide["success_only"]
        )
        for (subset, target), idx in wide.groupby(
            level=["evaluation_subset", "target"]
        ).groups.items():
            values = delta.loc[idx].to_numpy(dtype=float)
            try:
                p = wilcoxon(values).pvalue if len(values) >= 5 and np.any(values) else np.nan
            except ValueError:
                p = np.nan
            rows.append({
                "evaluation_subset": subset, "target": target, "metric": metric,
                "positive_means_success_only_better": True,
                "paired_difference_mean": np.mean(values),
                "paired_difference_std": np.std(values, ddof=1),
                "n_paired_folds": len(values), "wilcoxon_p": p,
            })
    return pd.DataFrame(rows)


def plot_primary_comparison(summary: pd.DataFrame, output_dir: Path):
    plot_df = summary.loc[
        (summary["evaluation_subset"] == "success_1")
        & (summary["component"] == "Ensemble_0.5_0.5")
    ].copy()
    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    palette = {"all_rows": "#5B8FF9", "success_only": "#E76F51"}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.1))
    specs = [
        ("r2", r"Outer-CV $R^2$", "(a) Goodness of fit"),
        ("mae", "Outer-CV MAE", "(b) Absolute error"),
        ("rmse", "Outer-CV RMSE", "(c) Root-mean-square error"),
    ]
    x = np.arange(len(TARGETS))
    width = 0.34
    labels = [r"$E_{x,\max}$", r"$E_{m,\max}$", "QY"]
    for ax, (metric, ylabel, title) in zip(axes, specs):
        for offset_i, policy in enumerate(POLICIES):
            g = plot_df.set_index(["policy", "target"])
            means = [g.loc[(policy, t), f"{metric}_mean"] for t in TARGETS]
            stds = [g.loc[(policy, t), f"{metric}_std"] for t in TARGETS]
            offset = (-0.5 if offset_i == 0 else 0.5) * width
            ax.bar(
                x + offset, means, width=width, yerr=stds, capsize=4,
                color=palette[policy], edgecolor="black", linewidth=0.7,
                label="All training rows" if policy == "all_rows" else "Success-only training",
            )
        ax.axhline(0, color="0.25", lw=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontweight="bold")
        sns.despine(ax=ax)
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Effect of excluding unsuccessful experiments from property-model training\n"
        "Primary evaluation: identical outer-test success=1 samples",
        fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "Fig_success_filter_comparison.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "Fig_success_filter_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = Cfg()
    if cfg.n_repeats > len(cfg.split_seeds):
        raise ValueError("Provide at least n_repeats split_seeds.")
    set_seed(cfg.random_seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prod = import_production_module(cfg.production_02_script)
    data = load_modeling_data(cfg, prod)
    print("\nDataset success counts:")
    print(data["success"].value_counts().sort_index())
    data.to_csv(output_dir / "dataset_used.csv", index=False, encoding="utf-8-sig")

    print("\nGenerating the fixed CatBoost feature table once...")
    feature_df = build_catboost_table(data, cfg, prod)
    splits = build_outer_splits(data, cfg)
    all_metrics, all_predictions, manifest = [], [], []

    for split_no, (repeat, fold, split_seed, tr, te) in enumerate(splits, start=1):
        test_success = int(data.iloc[te]["success"].sum())
        print(
            f"\n[Outer split {split_no}/{len(splits)}] repeat={repeat}, fold={fold}; "
            f"train={len(tr)}, test={len(te)}, test success=1: {test_success}"
        )
        manifest.append({
            "repeat": repeat, "fold": fold, "split_seed": split_seed,
            "n_train": len(tr), "n_test": len(te),
            "n_test_success_1": test_success,
            "train_row_ids": ";".join(map(str, data.iloc[tr]["row_id"])),
            "test_row_ids": ";".join(map(str, data.iloc[te]["row_id"])),
        })
        for policy in POLICIES:
            print(f"  Training policy: {policy}")
            results = train_predict_policy(
                prod, cfg, policy, data, feature_df, tr, te, split_seed
            )
            metric_rows, prediction_rows = collect_results(
                repeat, fold, split_seed, policy, *results
            )
            all_metrics.extend(metric_rows)
            all_predictions.extend(prediction_rows)

    fold_metrics = pd.DataFrame(all_metrics)
    predictions = pd.DataFrame(all_predictions)
    summary = summarize_metrics(fold_metrics)
    paired = paired_differences(fold_metrics)

    pd.DataFrame(manifest).to_csv(
        output_dir / "split_manifest.csv", index=False, encoding="utf-8-sig"
    )
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(
        output_dir / "out_of_fold_predictions.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(output_dir / "metric_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(
        output_dir / "paired_policy_differences.csv", index=False, encoding="utf-8-sig"
    )
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)
    plot_primary_comparison(summary, output_dir)

    print("\nPrimary result: 0.5FPNN-0.5CatBoost on identical success=1 test rows")
    cols = [
        "policy", "target", "r2_mean", "r2_std", "mae_mean", "mae_std",
        "rmse_mean", "rmse_std", "n_model_train_mean",
    ]
    primary = summary.loc[
        (summary["evaluation_subset"] == "success_1")
        & (summary["component"] == "Ensemble_0.5_0.5"), cols
    ]
    print(primary.to_string(index=False))
    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
