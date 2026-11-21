#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FP-encoder width/depth ablation for the revised 0.5 FPNN + 0.5 CatBoost model.

The current production encoder (679 -> 128 -> 32) is compared with shallower,
narrower, wider and deeper alternatives. Everything else is held fixed:

* the success-only property-model training domain;
* repeated outer CV splits;
* 14 -> 16 process encoder;
* symmetric precursor fusion [z1 + z2, |z1 - z2|];
* regression head and optimizer;
* selected CatBoost features/hyperparameters;
* 0.5 FPNN + 0.5 CatBoost prediction rule.

Required files
--------------
./02_train_fp_catboost_cycle_success_selected_features.py
./compare_success_filter_fpnn_catboost.py
./data/output_with_molar_amount_pubchem.xlsx
./data/observed_history.xlsx
./data/compound_smiles.xlsx
./cd_catboost_feature_selection/*_feature_list.csv

Run
---
python fp_encoder_dimension_ablation_ensemble.py

For a quick code check, set model_seeds=(42,) and n_repeats=1. For the final
manuscript analysis, retain the five model seeds and 3 x 5 repeated outer CV.
"""

from __future__ import annotations

# Allow the documented "python scripts/..." entry point from a checkout.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from catboost import CatBoostRegressor

# This helper is supplied with this script and reuses the production 02 code.
from validation import compare_success_filter_fpnn_catboost as base

warnings.filterwarnings("ignore")

TARGETS = ["Ex_max_nm", "Em_max_nm", "QY_%"]
CURRENT_ARCHITECTURE = "hidden128_latent32"

# Each tuple contains the hidden layer widths followed by the latent width.
# Example: (128, 32) means 679 -> 128 -> 32.
FP_ARCHITECTURES = {
    "linear_latent32": (32,),
    "hidden64_latent32": (64, 32),
    "hidden128_latent16": (128, 16),
    "hidden128_latent32": (128, 32),       # current model
    "hidden128_latent64": (128, 64),
    "hidden256_latent32": (256, 32),
    "deep128_64_latent32": (128, 64, 32),
    "deep256_128_latent32": (256, 128, 32),
}


@dataclass
class Config(base.Cfg):
    observed_history_file: str = ""
    output_dir: str = "./outputs/fingerprint_encoder"

    # Manuscript setting: 3 repetitions x 5 outer folds and five model seeds.
    n_repeats: int = 3
    n_splits: int = 5
    split_seeds: tuple = (42, 2024, 3407)
    model_seeds: tuple = (42, 2024, 3407, 777, 1024)

    # Same training settings as the revised production FPNN.
    nn_epochs: int = 300
    nn_lr: float = 2e-3
    nn_weight_decay: float = 1e-4
    nn_batch_size: int = 64
    nn_grad_clip: float = 1.0
    fp_dropout: float = 0.30
    process_dropout: float = 0.10
    fusion_dropout: float = 0.30

    # Fixed ensemble rule.
    ensemble_fpnn_weight: float = 0.5
    ensemble_catboost_weight: float = 0.5


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConfigurableFingerprintEncoder(nn.Module):
    """MLP encoder whose final width is the molecular latent dimension."""

    def __init__(self, widths: tuple[int, ...], input_dim: int = 679,
                 dropout: float = 0.30):
        super().__init__()
        if not widths or any(int(w) <= 0 for w in widths):
            raise ValueError(f"Invalid FP architecture: {widths}")
        layers = []
        previous = input_dim
        for i, width in enumerate(widths):
            layers.extend([nn.Linear(previous, int(width)), nn.ReLU()])
            # Match production behavior: dropout follows hidden layers, not output.
            if i < len(widths) - 1:
                layers.append(nn.Dropout(dropout))
            previous = int(width)
        self.net = nn.Sequential(*layers)
        self.latent_dim = int(widths[-1])

    def forward(self, x):
        return self.net(x.float())


class FPNNVariant(nn.Module):
    """Production FPNN with only the FP encoder architecture made configurable."""

    def __init__(self, widths: tuple[int, ...], cfg: Config):
        super().__init__()
        self.fp_encoder = ConfigurableFingerprintEncoder(
            widths=widths, input_dim=679, dropout=cfg.fp_dropout
        )
        latent = self.fp_encoder.latent_dim
        self.proc_encoder = nn.Sequential(
            nn.Linear(14, 16), nn.ReLU(), nn.Dropout(cfg.process_dropout)
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * latent, 64), nn.GELU(),
            nn.Dropout(cfg.fusion_dropout),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(80, 64), nn.GELU(),
            nn.Dropout(cfg.fusion_dropout), nn.Linear(64, 3),
        )

    def forward(self, fps_1, fps_2, global_feat):
        z1 = self.fp_encoder(fps_1)
        z2 = self.fp_encoder(fps_2)
        pair = torch.cat([z1 + z2, torch.abs(z1 - z2)], dim=-1)
        fusion = self.fusion_mlp(pair)
        process = self.proc_encoder(global_feat)
        return self.head(torch.cat([fusion, process], dim=-1))


def train_fpnn_variant(prod, architecture, train_df, scaler, cfg, seed, device):
    seed_all(seed)
    smiles = pd.read_excel(cfg.smiles_path)
    featurizer = prod.CarbonDotFeaturizer(
        smiles, prod.get_solvent_polarity_data()
    )
    dataset = prod.FPProcessDataset(train_df, featurizer, has_y=True)
    loader = DataLoader(
        dataset, batch_size=cfg.nn_batch_size, shuffle=True,
        collate_fn=prod.collate_fn, num_workers=0,
    )
    model = FPNNVariant(FP_ARCHITECTURES[architecture], cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.nn_lr, weight_decay=cfg.nn_weight_decay
    )
    for _ in range(cfg.nn_epochs):
        model.train()
        for batch in loader:
            scaled_y = scaler.transform(
                batch["y"].numpy().astype(np.float32)
            ).astype(np.float32)
            prediction = model(
                batch["fps_1"].to(device), batch["fps_2"].to(device),
                batch["global_feat"].to(device),
            )
            target = torch.as_tensor(scaled_y, dtype=torch.float32, device=device)
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.nn_grad_clip)
            optimizer.step()
    return model, featurizer


@torch.no_grad()
def predict_fpnn(prod, model, featurizer, df, scaler, cfg, device):
    dataset = prod.FPProcessDataset(df, featurizer, has_y=False)
    loader = DataLoader(
        dataset, batch_size=cfg.nn_batch_size, shuffle=False,
        collate_fn=prod.collate_fn, num_workers=0,
    )
    chunks = []
    model.eval()
    for batch in loader:
        scaled = model(
            batch["fps_1"].to(device), batch["fps_2"].to(device),
            batch["global_feat"].to(device),
        ).cpu().numpy()
        chunks.append(scaler.inverse_transform(scaled))
    return np.vstack(chunks)


def selected_features(prod, feature_df):
    result = {}
    for target in TARGETS:
        result[target] = prod.read_feature_list(prod.FEATURE_LIST_FILES[target])
        missing = [x for x in result[target] if x not in feature_df.columns]
        if missing:
            raise ValueError(f"Missing CatBoost features for {target}: {missing[:10]}")
    return result


def train_catboost_cache(prod, cfg, feature_df, model_train, test_df,
                         split_seed, selected):
    """Train CatBoost once per outer fold; reuse it for every FPNN variant."""
    indexed = feature_df.set_index("row_id")
    train_features = indexed.loc[model_train["row_id"]].reset_index()
    test_features = indexed.loc[test_df["row_id"]].reset_index()
    train_by_seed, test_by_seed = [], []
    for model_i, base_seed in enumerate(cfg.model_seeds):
        seed = int(base_seed + 10000 * split_seed + 100 * model_i)
        pred_train_targets, pred_test_targets = [], []
        for target in TARGETS:
            features = selected[target]
            model = CatBoostRegressor(
                loss_function="RMSE", iterations=cfg.cb_iterations,
                learning_rate=cfg.cb_learning_rate, depth=cfg.cb_depth,
                l2_leaf_reg=cfg.cb_l2_leaf_reg,
                random_strength=cfg.cb_random_strength,
                bootstrap_type="Bernoulli", subsample=cfg.cb_subsample,
                grow_policy="SymmetricTree", random_seed=seed,
                verbose=False, allow_writing_files=False, thread_count=-1,
            )
            model.fit(
                base.catboost_matrix(train_features, features),
                train_features[target].astype(float), verbose=False,
            )
            pred_train_targets.append(model.predict(
                base.catboost_matrix(train_features, features)
            ))
            pred_test_targets.append(model.predict(
                base.catboost_matrix(test_features, features)
            ))
        train_by_seed.append(np.column_stack(pred_train_targets))
        test_by_seed.append(np.column_stack(pred_test_targets))
    return (
        np.mean(np.stack(train_by_seed), axis=0),
        np.mean(np.stack(test_by_seed), axis=0),
    )


def add_metric_rows(rows, repeat, fold, architecture, component, split,
                    y_true, y_pred, parameter_count):
    target_r2 = []
    for j, target in enumerate(TARGETS):
        r2 = r2_score(y_true[:, j], y_pred[:, j])
        target_r2.append(r2)
        rows.append({
            "repeat": repeat, "fold": fold, "architecture": architecture,
            "current_architecture": architecture == CURRENT_ARCHITECTURE,
            "component": component, "split": split, "target": target,
            "n": len(y_true), "parameter_count": parameter_count,
            "r2": r2,
            "mae": mean_absolute_error(y_true[:, j], y_pred[:, j]),
            "rmse": mean_squared_error(y_true[:, j], y_pred[:, j]) ** 0.5,
        })
    rows.append({
        "repeat": repeat, "fold": fold, "architecture": architecture,
        "current_architecture": architecture == CURRENT_ARCHITECTURE,
        "component": component, "split": split, "target": "Macro",
        "n": len(y_true), "parameter_count": parameter_count,
        "r2": float(np.mean(target_r2)), "mae": np.nan, "rmse": np.nan,
    })


def summarize(fold_df):
    group = ["architecture", "current_architecture", "component", "split", "target"]
    summary = fold_df.groupby(group, as_index=False).agg(
        n_outer_folds=("r2", "size"), parameter_count=("parameter_count", "first"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
    )
    return summary


def architecture_ranking(summary):
    macro = summary.query(
        "component == 'Ensemble_0.5_0.5' and target == 'Macro'"
    ).pivot(index="architecture", columns="split", values="r2_mean")
    test_sd = summary.query(
        "component == 'Ensemble_0.5_0.5' and target == 'Macro' and split == 'test'"
    ).set_index("architecture")["r2_std"]
    params = summary.query(
        "component == 'Ensemble_0.5_0.5' and target == 'Macro' and split == 'test'"
    ).set_index("architecture")["parameter_count"]
    out = pd.DataFrame({
        "test_macro_r2_mean": macro["test"],
        "test_macro_r2_std": test_sd,
        "train_macro_r2_mean": macro["train"],
        "train_test_gap": macro["train"] - macro["test"],
        "parameter_count": params,
    }).reset_index()
    out["current_architecture"] = out["architecture"].eq(CURRENT_ARCHITECTURE)
    return out.sort_values("test_macro_r2_mean", ascending=False).reset_index(drop=True)


def make_figure(summary, ranking, output_dir):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    order = ranking.sort_values("test_macro_r2_mean")["architecture"].tolist()
    colors = ["#D95F02" if x == CURRENT_ARCHITECTURE else "#6B93C6" for x in order]

    plot = ranking.set_index("architecture").loc[order]
    axes[0].barh(
        np.arange(len(order)), plot["test_macro_r2_mean"],
        xerr=plot["test_macro_r2_std"], color=colors, edgecolor="black",
        linewidth=0.6, capsize=3,
    )
    axes[0].set_yticks(np.arange(len(order)), order)
    axes[0].set_xlabel(r"Outer-CV macro-mean $R^2$")
    axes[0].set_title("(a) Ensemble performance", loc="left", fontweight="bold")

    for _, row in ranking.iterrows():
        color = "#D95F02" if row["current_architecture"] else "#4C78A8"
        axes[1].scatter(
            row["parameter_count"], row["test_macro_r2_mean"],
            s=85 if row["current_architecture"] else 48,
            c=color, edgecolor="black", linewidth=0.5, zorder=3,
        )
        axes[1].annotate(row["architecture"],
                         (row["parameter_count"], row["test_macro_r2_mean"]),
                         xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Trainable FPNN parameters (log scale)")
    axes[1].set_ylabel(r"Outer-CV macro-mean $R^2$")
    axes[1].set_title("(b) Accuracy-complexity trade-off", loc="left", fontweight="bold")

    heat = summary.query(
        "component == 'Ensemble_0.5_0.5' and split == 'test' and target != 'Macro'"
    ).pivot(index="architecture", columns="target", values="r2_mean")
    heat = heat.reindex(ranking["architecture"])[TARGETS]
    sns.heatmap(
        heat, ax=axes[2], cmap="Blues", annot=True, fmt=".2f",
        linewidths=0.6, cbar_kws={"label": r"Outer-CV $R^2$"},
    )
    axes[2].set_xlabel("")
    axes[2].set_ylabel("")
    axes[2].set_title("(c) Target-resolved performance", loc="left", fontweight="bold")
    axes[2].set_xticklabels([r"$E_{x,max}$", r"$E_{m,max}$", "QY"], rotation=0)

    fig.suptitle(
        "Fingerprint-encoder ablation in the 0.5 FPNN–0.5 CatBoost ensemble",
        fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "Fig_fp_encoder_dimension_ablation.png",
                dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "Fig_fp_encoder_dimension_ablation.pdf",
                bbox_inches="tight")
    plt.close(fig)


def main(cfg=None):
    cfg = cfg or Config()
    if cfg.n_repeats > len(cfg.split_seeds):
        raise ValueError("Provide at least n_repeats split seeds.")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), "architectures": FP_ARCHITECTURES},
                  f, ensure_ascii=False, indent=2)

    prod = base.import_production_module(cfg.production_02_script)
    all_data = base.load_modeling_data(cfg, prod)
    feature_df = base.build_catboost_table(all_data, cfg, prod)
    selected = selected_features(prod, feature_df)
    splits = base.build_outer_splits(all_data, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, prediction_rows = [], []

    print("Dataset success counts:")
    print(all_data["success"].value_counts().sort_index())
    print(f"Architectures: {len(FP_ARCHITECTURES)}; outer folds: {len(splits)}")

    for repeat, fold, split_seed, train_idx, test_idx in splits:
        outer_train = all_data.iloc[train_idx].copy()
        outer_test = all_data.iloc[test_idx].copy()
        model_train = outer_train.loc[outer_train["success"] == 1].copy()
        model_test = outer_test.loc[outer_test["success"] == 1].copy()
        if min(len(model_train), len(model_test)) < 3:
            raise RuntimeError("Too few success=1 rows in a fold.")

        scaler = StandardScaler().fit(model_train[TARGETS].to_numpy(float))
        cb_train, cb_test = train_catboost_cache(
            prod, cfg, feature_df, model_train, model_test,
            split_seed, selected,
        )

        for architecture in FP_ARCHITECTURES:
            fp_train_seeds, fp_test_seeds = [], []
            parameter_count = None
            for model_i, base_seed in enumerate(cfg.model_seeds):
                seed = int(base_seed + 10000 * split_seed + 100 * model_i)
                model, featurizer = train_fpnn_variant(
                    prod, architecture, model_train, scaler, cfg, seed, device
                )
                if parameter_count is None:
                    parameter_count = sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    )
                fp_train_seeds.append(predict_fpnn(
                    prod, model, featurizer, model_train, scaler, cfg, device
                ))
                fp_test_seeds.append(predict_fpnn(
                    prod, model, featurizer, model_test, scaler, cfg, device
                ))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            fp_train = np.mean(np.stack(fp_train_seeds), axis=0)
            fp_test = np.mean(np.stack(fp_test_seeds), axis=0)
            ensemble_train = 0.5 * fp_train + 0.5 * cb_train
            ensemble_test = 0.5 * fp_test + 0.5 * cb_test
            true_train = model_train[TARGETS].to_numpy(float)
            true_test = model_test[TARGETS].to_numpy(float)

            for component, train_pred, test_pred in [
                ("FPNN", fp_train, fp_test),
                ("Ensemble_0.5_0.5", ensemble_train, ensemble_test),
            ]:
                add_metric_rows(rows, repeat, fold, architecture, component,
                                "train", true_train, train_pred, parameter_count)
                add_metric_rows(rows, repeat, fold, architecture, component,
                                "test", true_test, test_pred, parameter_count)

            for i, (_, sample) in enumerate(model_test.iterrows()):
                for j, target in enumerate(TARGETS):
                    prediction_rows.append({
                        "repeat": repeat, "fold": fold,
                        "architecture": architecture,
                        "row_id": int(sample["row_id"]), "target": target,
                        "y_true": float(sample[target]),
                        "pred_fpnn": float(fp_test[i, j]),
                        "pred_catboost": float(cb_test[i, j]),
                        "pred_ensemble": float(ensemble_test[i, j]),
                    })
            print(f"repeat={repeat}, fold={fold}, architecture={architecture} finished")

    fold_df = pd.DataFrame(rows)
    summary = summarize(fold_df)
    ranking = architecture_ranking(summary)
    predictions = pd.DataFrame(prediction_rows)
    fold_df.to_csv(output_dir / "fp_encoder_fold_metrics.csv", index=False,
                   encoding="utf-8-sig")
    summary.to_csv(output_dir / "fp_encoder_metric_summary.csv", index=False,
                   encoding="utf-8-sig")
    ranking.to_csv(output_dir / "fp_encoder_architecture_ranking.csv", index=False,
                   encoding="utf-8-sig")
    predictions.to_csv(output_dir / "fp_encoder_oof_predictions.csv", index=False,
                       encoding="utf-8-sig")
    make_figure(summary, ranking, output_dir)

    print("\nArchitecture ranking (0.5/0.5 ensemble):")
    print(ranking.to_string(index=False))
    print(f"\nAll outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate eight fingerprint encoders with matched splits.')
    parser.add_argument("--history", type=Path, help="Include a local recipe-level history in this retrospective analysis.")
    parser.add_argument("--quick", action="store_true", help="Exercise the analysis with two folds and a reduced fit budget.")
    args = parser.parse_args()
    cfg = Config()
    if args.history:
        if not args.history.is_file():
            parser.error(f"History file does not exist: {args.history}")
        cfg.observed_history_file = str(args.history)
    if args.quick:
        cfg.n_splits = 2
        cfg.n_repeats = 1
        cfg.nn_epochs = 1
        cfg.cb_iterations = 5
        cfg.split_seeds = (42,)
        cfg.model_seeds = (42,)
        cfg.output_dir = "outputs/quick/fingerprint_encoder"
    main(cfg)
