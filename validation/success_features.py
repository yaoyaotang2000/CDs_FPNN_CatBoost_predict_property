# independent_success_feature_selection.py
# -*- coding: utf-8 -*-
"""
Independent feature selection for the P_success classifier.

This script is independent from the active-learning / BO loop. It reads an
existing success-classifier feature table, performs repeated stratified CV,
ranks CatBoost feature importance, compares All/Top-K feature subsets, and
saves a compact selected feature list.

Run:
    python independent_success_feature_selection.py

Before running, modify Cfg.success_feature_table and Cfg.output_dir if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


@dataclass
class Cfg:
    # Existing engineered feature table containing a success column.
    success_feature_table: str = "./feature_inputs/initial_features.csv"

    # Independent output folder, not tied to a closed-loop run directory.
    output_dir: str = "./success_feature_selection_independent"

    success_col: str = "success"
    success_positive_threshold: float = 0.5

    # Feature subset sizes to compare.
    top_k_list: tuple = (10, 15, 20, 25, 30)

    # Repeated stratified CV.
    n_splits: int = 5
    n_repeats: int = 5
    random_seed: int = 42

    # CatBoost parameters.
    iterations: int = 1500
    learning_rate: float = 0.04
    depth: int = 4
    l2_leaf_reg: float = 8.0
    random_strength: float = 1.0
    subsample: float = 0.9

    # Metrics threshold.
    classification_threshold: float = 0.5

    # Selection rule: choose smallest subset within this AUC gap from best.
    auc_tolerance: float = 0.01
    min_top_k: int = 20

    # Optional final selected-feature models.
    train_final_models: bool = True
    final_model_seeds: tuple = (42, 2024, 3407, 777, 1024)

    dpi: int = 300


TARGET_OR_LEAKAGE_COLS = {
    "Ex_max_nm", "Em_max_nm", "QY_%",
    "stoke_shift", "stokes_shift", "Stokes_shift_nm", "stoke_shift_calc_check",
    "success",
}

META_COLS = {
    "candidate_id", "experiment_id", "cycle", "target_color", "selection_type",
    "data_source", "recipe_key", "notes",
    "precursor_1", "precursor_2", "reagent", "reagent_1",
    "precursor_1_key", "precursor_2_key", "reagent_1_key",
    "p1_compound_name", "p1_compound_key", "p1_SMILES",
    "p2_compound_name", "p2_compound_key", "p2_SMILES",
    "compound_name", "compound_key", "SMILES",
}


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if str(p).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(p)
    return pd.read_csv(p)


def ensure_dirs(cfg: Cfg):
    out_dir = Path(cfg.output_dir)
    fig_dir = out_dir / "figures"
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, fig_dir, model_dir


def infer_candidate_features(df: pd.DataFrame, cfg: Cfg) -> list[str]:
    exclude = set(TARGET_OR_LEAKAGE_COLS) | set(META_COLS) | {cfg.success_col}
    features = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith("pred_") or col.startswith("observed_"):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if not df[col].notna().any():
            continue
        features.append(col)
    if not features:
        raise ValueError("No numeric non-leakage features were inferred.")
    return features


def prepare_xy(df: pd.DataFrame, features: list[str], cfg: Cfg):
    if cfg.success_col not in df.columns:
        raise ValueError(f"Required success label column not found: {cfg.success_col}")
    data = df.dropna(subset=[cfg.success_col]).copy()
    y_raw = pd.to_numeric(data[cfg.success_col], errors="coerce")
    data = data.loc[y_raw.notna()].copy()
    y_raw = y_raw.loc[y_raw.notna()]
    y = (y_raw.astype(float) >= cfg.success_positive_threshold).astype(int)
    if y.nunique() < 2:
        raise ValueError(f"Need both success=0 and success=1 labels. Got: {y.value_counts().to_dict()}")
    X = data[features].copy()
    for col in features:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X, y


def make_catboost(seed: int, cfg: Cfg) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=cfg.iterations,
        learning_rate=cfg.learning_rate,
        depth=cfg.depth,
        l2_leaf_reg=cfg.l2_leaf_reg,
        random_strength=cfg.random_strength,
        bootstrap_type="Bernoulli",
        subsample=cfg.subsample,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def make_cv(y: pd.Series, cfg: Cfg):
    min_class = int(y.value_counts().min())
    n_splits = min(cfg.n_splits, min_class)
    if n_splits < 2:
        raise ValueError(f"Too few minority-class samples for CV: {y.value_counts().to_dict()}")
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=cfg.n_repeats, random_state=cfg.random_seed)


def safe_auc(y_true, y_prob):
    return np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_prob)


def safe_pr_auc(y_true, y_prob):
    return np.nan if len(np.unique(y_true)) < 2 else average_precision_score(y_true, y_prob)


def estimate_cv_feature_importance(X: pd.DataFrame, y: pd.Series, features: list[str], cfg: Cfg) -> pd.DataFrame:
    cv = make_cv(y, cfg)
    records = []
    for fold_idx, (tr_idx, _) in enumerate(cv.split(X[features], y), start=1):
        Xtr = X.iloc[tr_idx][features]
        ytr = y.iloc[tr_idx]
        model = make_catboost(seed=cfg.random_seed + 10000 + fold_idx, cfg=cfg)
        model.fit(Xtr, ytr, verbose=False)
        importance = model.get_feature_importance(type="PredictionValuesChange")
        for feat, imp in zip(features, importance):
            records.append({"fold": fold_idx, "feature": feat, "importance": float(imp)})
        print(f"Feature-importance CV fold {fold_idx} finished.")
    imp_df = pd.DataFrame(records)
    rank_df = (
        imp_df.groupby("feature")["importance"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "importance_mean", "std": "importance_std"})
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    rank_df.insert(0, "rank", np.arange(1, len(rank_df) + 1))
    return rank_df


def evaluate_subset(X: pd.DataFrame, y: pd.Series, features: list[str], subset_name: str, cfg: Cfg) -> pd.DataFrame:
    cv = make_cv(y, cfg)
    rows = []
    for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X[features], y), start=1):
        Xtr, Xte = X.iloc[tr_idx][features], X.iloc[te_idx][features]
        ytr, yte = y.iloc[tr_idx], y.iloc[te_idx]
        model = make_catboost(seed=cfg.random_seed + fold_idx, cfg=cfg)
        model.fit(Xtr, ytr, verbose=False)
        prob = model.predict_proba(Xte)[:, 1]
        pred = (prob >= cfg.classification_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred, labels=[0, 1]).ravel()
        rows.append({
            "subset": subset_name,
            "n_features": len(features),
            "fold": fold_idx,
            "auc": safe_auc(yte, prob),
            "pr_auc": safe_pr_auc(yte, prob),
            "balanced_accuracy": balanced_accuracy_score(yte, pred),
            "accuracy": accuracy_score(yte, pred),
            "f1": f1_score(yte, pred, zero_division=0),
            "precision": precision_score(yte, pred, zero_division=0),
            "recall_success_1": recall_score(yte, pred, pos_label=1, zero_division=0),
            "recall_failure_0": recall_score(yte, pred, pos_label=0, zero_division=0),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        })
    return pd.DataFrame(rows)


def aggregate_cv_metrics(cv_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["auc", "pr_auc", "balanced_accuracy", "accuracy", "f1", "precision", "recall_success_1", "recall_failure_0"]
    rows = []
    for subset, sub in cv_df.groupby("subset"):
        row = {"subset": subset, "n_features": int(sub["n_features"].iloc[0])}
        for metric in metrics:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["auc_mean", "balanced_accuracy_mean"], ascending=False).reset_index(drop=True)


def choose_compact_subset(summary: pd.DataFrame, cfg: Cfg) -> dict:
    best_auc = summary["auc_mean"].max()
    candidates = summary[summary["auc_mean"] >= best_auc - cfg.auc_tolerance].copy()
    candidates = candidates[candidates["n_features"] >= cfg.min_top_k].copy()
    if len(candidates) == 0:
        candidates = summary.copy()
    candidates = candidates.sort_values(["n_features", "balanced_accuracy_mean", "f1_mean"], ascending=[True, False, False])
    return candidates.iloc[0].to_dict()


def train_final_models(X: pd.DataFrame, y: pd.Series, selected_features: list[str], cfg: Cfg, model_dir: Path, out_dir: Path):
    if not cfg.train_final_models:
        return
    rows = []
    for i, seed in enumerate(cfg.final_model_seeds):
        print(f"Training final selected-feature classifier, seed={seed}")
        model = make_catboost(seed=seed + 2000 * i, cfg=cfg)
        model.fit(X[selected_features], y, verbose=False)
        model_path = model_dir / f"catboost_success_selected_seed{seed}.cbm"
        model.save_model(str(model_path))
        rows.append({"seed": seed, "model_path": str(model_path), "n_features": len(selected_features)})
    pd.DataFrame(rows).to_csv(out_dir / "success_selected_final_model_summary.csv", index=False, encoding="utf-8-sig")


def plot_feature_importance(rank_df: pd.DataFrame, fig_dir: Path, cfg: Cfg, top_n: int = 30):
    top = rank_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, max(5, 0.28 * len(top))))
    plt.barh(top["feature"], top["importance_mean"])
    plt.xlabel("Mean CatBoost importance")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} P_success features")
    plt.tight_layout()
    plt.savefig(fig_dir / "success_feature_importance_top30.png", dpi=cfg.dpi)
    plt.close()


def plot_cv_summary(summary: pd.DataFrame, fig_dir: Path, cfg: Cfg):
    tmp = summary.sort_values("n_features")
    plt.figure(figsize=(7, 5))
    plt.errorbar(tmp["n_features"], tmp["auc_mean"], yerr=tmp["auc_std"], marker="o", capsize=3)
    plt.xlabel("Number of features")
    plt.ylabel("CV AUC")
    plt.title("P_success feature subset selection")
    plt.tight_layout()
    plt.savefig(fig_dir / "success_feature_selection_auc.png", dpi=cfg.dpi)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.errorbar(tmp["n_features"], tmp["balanced_accuracy_mean"], yerr=tmp["balanced_accuracy_std"], marker="o", capsize=3)
    plt.xlabel("Number of features")
    plt.ylabel("CV balanced accuracy")
    plt.title("P_success feature subset selection")
    plt.tight_layout()
    plt.savefig(fig_dir / "success_feature_selection_balanced_accuracy.png", dpi=cfg.dpi)
    plt.close()


def main():
    cfg = Cfg()
    out_dir, fig_dir, model_dir = ensure_dirs(cfg)

    print(f"Loading feature table: {cfg.success_feature_table}")
    df = read_table(cfg.success_feature_table)

    all_features = infer_candidate_features(df, cfg)
    X, y = prepare_xy(df, all_features, cfg)

    print("\nSuccess label counts:")
    print(y.value_counts())
    print(f"Rows with labels: {len(y)}")
    print(f"Candidate features before selection: {len(all_features)}")

    pd.DataFrame({"feature": all_features}).to_csv(out_dir / "success_all_inferred_feature_list.csv", index=False, encoding="utf-8-sig")

    print("\nEstimating repeated-CV feature importance...")
    importance_rank = estimate_cv_feature_importance(X, y, all_features, cfg)
    importance_rank.to_csv(out_dir / "success_feature_importance_cv.csv", index=False, encoding="utf-8-sig")

    subset_map = {"all": all_features}
    for k in cfg.top_k_list:
        k = min(int(k), len(all_features))
        subset_map[f"top{k}"] = importance_rank.head(k)["feature"].tolist()

    print("\nEvaluating feature subsets...")
    cv_results = []
    for subset_name, features in subset_map.items():
        print(f"Evaluating {subset_name}: n_features={len(features)}")
        cv_results.append(evaluate_subset(X, y, features, subset_name, cfg))

    cv_all = pd.concat(cv_results, ignore_index=True)
    cv_all.to_csv(out_dir / "success_feature_selection_cv_results.csv", index=False, encoding="utf-8-sig")

    summary = aggregate_cv_metrics(cv_all)
    summary.to_csv(out_dir / "success_feature_selection_summary.csv", index=False, encoding="utf-8-sig")

    selected_info = choose_compact_subset(summary, cfg)
    selected_subset_name = selected_info["subset"]
    selected_features = subset_map[selected_subset_name]

    selected_file = out_dir / "success_classifier_feature_list_selected.csv"
    pd.DataFrame({"feature": selected_features}).to_csv(selected_file, index=False, encoding="utf-8-sig")

    with open(out_dir / "success_feature_selection_config.json", "w", encoding="utf-8") as f:
        json.dump({
            **asdict(cfg),
            "n_labeled_rows": int(len(y)),
            "label_counts": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
            "n_all_features": int(len(all_features)),
            "selected_subset": selected_subset_name,
            "selected_n_features": int(len(selected_features)),
            "selected_metrics": selected_info,
        }, f, ensure_ascii=False, indent=2)

    print("\nSelected feature subset:")
    print(f"  subset: {selected_subset_name}")
    print(f"  n_features: {len(selected_features)}")
    for k, v in selected_info.items():
        print(f"  {k}: {v}")

    print("\nTraining final selected-feature classifier ensemble...")
    train_final_models(X, y, selected_features, cfg, model_dir, out_dir)

    plot_feature_importance(importance_rank, fig_dir, cfg, top_n=30)
    plot_cv_summary(summary, fig_dir, cfg)

    print("\nDone.")
    print(f"Selected feature list: {selected_file}")
    print(f"All results saved to: {Path(cfg.output_dir).resolve()}")


if __name__ == "__main__":
    main()
