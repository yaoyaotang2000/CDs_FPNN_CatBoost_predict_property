#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FPNN process-encoder ablation and fixed-test learning-curve analysis.

Purpose
-------
1. Compare process-encoder width/depth while keeping the fingerprint encoder,
   precursor fusion, regression head, data splits, optimizer, and training rule fixed.
2. Evaluate data-size dependence using a fixed outer test fold. Only the outer
   training fold is subsampled, so the test set does not change with train fraction.

Data
----
./data/output_with_molar_amount_pubchem.xlsx  initial experimental data
./data/observed_history.xlsx                  34 prospective observations

The script reuses CarbonDotFeaturizer, data_process, and data-cleaning helpers
from the production training script. Set Config.production_script to the actual
path of 02_train_fp_catboost_cycle.py.

Outputs
-------
./fp_process_ablation_results/
    dataset_manifest.json
    process_encoder_fold_metrics.csv
    process_encoder_summary.csv
    learning_curve_fold_metrics.csv
    learning_curve_summary.csv
    Fig_process_encoder_ablation.png
    Fig_fixed_test_learning_curve.png
    Fig_ablation_and_learning_curve.png

Run
---
python fp_process_encoder_ablation_learning_curve.py
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
import math
import random
import runpy
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


TARGETS = ["Ex_max_nm", "Em_max_nm", "QY_%"]

MOLECULAR_WEIGHTS = {
    "Salicylic acid": 138.12, "DL-Methionine": 149.21, "Glucose": 180.16,
    "2-Aminoterephthalic acid": 181.14, "L-Leucine": 131.17,
    "Dopamine hydrochloride": 189.64, "Urea": 60.06, "L-Phenylalanine": 165.19,
    "Polyaniline": 91.13, "p-Aminobenzenesulfonic acid": 173.19,
    "p-Phenylenediamine": 108.14, "L-Cystine": 240.30, "L-Cysteine": 121.16,
    "m-Phenylenediamine": 108.14, "o-Phenylenediamine": 108.14,
    "Neutral Red": 288.78, "Citric acid": 192.12, "Glycine": 75.07,
    "L-Lysine": 146.19, "L-Tyrosine": 181.19, "Melamine": 126.12,
    "p-Aminophenol": 109.13, "L-Serine": 105.09, "L-Threonine": 119.12,
    "Ethanolamine": 61.08, "Terephthalic acid": 166.13, "Thiourea": 76.12,
    "Tartaric acid": 150.09,
}


@dataclass
class Config:
    production_script: str = "./validation/02_train_fp_catboost_cycle_success_selected_features.py"
    initial_file: str = "./data/output_with_molar_amount_pubchem.xlsx"
    history_file: str = ""
    smiles_file: str = "./data/compound_smiles.xlsx"
    output_dir: str = "./outputs/process_encoder"

    # Same convention as the production property model.
    property_success_only: bool = True
    success_threshold: float = 0.50

    # Repeated 3 × 5 CV.
    n_splits: int = 5
    n_repeats: int = 3
    cv_seed: int = 42

    # Training.
    epochs: int = 300
    patience: int = 30
    min_delta: float = 1e-4
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    grad_clip: float = 1.0
    inner_val_fraction: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Learning curve. Fractions apply only to each outer training fold.
    train_fractions: tuple = (0.25, 0.50, 0.75, 1.00)

    # Full encoder grid. "linear16" is the current production structure.
    encoder_names: tuple = (
        "identity14",
        "linear8",
        "linear16",
        "linear32",
        "two_layer_32_16",
        "two_layer_64_16",
        "two_layer_128_32",
        "old_two_layer_256_256",
    )

    # Use the selected/current encoder for the learning curve.
    learning_curve_encoder: str = "linear16"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)


def load_production_namespace(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Production script not found: {p}. "
            "Set Config.production_script to 02_train_fp_catboost_cycle.py."
        )
    # run_name is not "__main__", so its main() function is not executed.
    ns = runpy.run_path(str(p), run_name="fp_production_module")
    required = [
        "CarbonDotFeaturizer", "data_process",
        "get_solvent_polarity_data", "ensure_shift_and_success_columns",
    ]
    missing = [name for name in required if name not in ns]
    if missing:
        raise AttributeError(f"Production script is missing definitions: {missing}")
    return ns


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "reagent": "reagent_1",
        "mass_1": "mass_1_g",
        "mass_2": "mass_2_g",
        "Mass1(g)": "mass_1_g",
        "Mass2(g)": "mass_2_g",
        "stokes_shift": "Stokes_shift_nm",
        "stoke_shift": "Stokes_shift_nm",
    }
    for old, new in rename.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    if "reagent_1_mL" not in out.columns:
        out["reagent_1_mL"] = 15.0
    for c in ["pre_1_mmol", "pre_2_mmol"]:
        if c not in out.columns:
            out[c] = np.nan

    for i, row in out.iterrows():
        for n, pcol, mcol, mmolcol in [
            (1, "precursor_1", "mass_1_g", "pre_1_mmol"),
            (2, "precursor_2", "mass_2_g", "pre_2_mmol"),
        ]:
            if pd.isna(row.get(mmolcol, np.nan)):
                precursor = str(row.get(pcol, ""))
                mass = row.get(mcol, np.nan)
                if precursor in MOLECULAR_WEIGHTS and pd.notna(mass):
                    out.loc[i, mmolcol] = float(mass) / MOLECULAR_WEIGHTS[precursor] * 1000.0

    if "Stokes_shift_nm" not in out.columns and {"Em_max_nm", "Ex_max_nm"}.issubset(out.columns):
        out["Stokes_shift_nm"] = (
            pd.to_numeric(out["Em_max_nm"], errors="coerce")
            - pd.to_numeric(out["Ex_max_nm"], errors="coerce")
        )

    if "success" not in out.columns:
        qy = pd.to_numeric(out.get("QY_%"), errors="coerce")
        shift = pd.to_numeric(out.get("Stokes_shift_nm"), errors="coerce")
        out["success"] = (~((qy < 4.0) & (shift < 80.0))).astype(float)
    else:
        out["success"] = pd.to_numeric(out["success"], errors="coerce")

    return out


def recipe_key(df: pd.DataFrame) -> pd.Series:
    keys = []
    for _, r in df.iterrows():
        pair = sorted([
            (str(r.get("precursor_1", "")).strip(), round(float(r.get("pre_1_mmol")), 6)),
            (str(r.get("precursor_2", "")).strip(), round(float(r.get("pre_2_mmol")), 6)),
        ])
        keys.append((
            pair[0], pair[1],
            str(r.get("reagent_1", "")).strip(),
            round(float(r.get("temperature")), 6),
            round(float(r.get("time_h")), 6),
        ))
    return pd.Series(keys, index=df.index)


def load_combined_property_data(cfg: Config, ns: dict) -> tuple[pd.DataFrame, dict]:
    initial = standardize_columns(read_table(cfg.initial_file))
    initial["data_source"] = "initial"

    history = pd.DataFrame()
    combined = initial
    if cfg.history_file:
        history = standardize_columns(read_table(cfg.history_file)).dropna(subset=TARGETS).copy()
        history["data_source"] = "observed_history"
        combined = pd.concat([initial, history], ignore_index=True, sort=False)
    combined = ns["ensure_shift_and_success_columns"](combined)
    combined = ns["data_process"](combined).reset_index(drop=True)
    combined = ns["ensure_shift_and_success_columns"](combined)

    required = [
        "precursor_1", "precursor_2", "reagent_1",
        "temperature", "time_h", "pre_1_mmol", "pre_2_mmol",
        *TARGETS,
    ]
    missing = [c for c in required if c not in combined.columns]
    if missing:
        raise ValueError(f"Combined data missing required columns: {missing}")

    combined = combined.dropna(subset=required).copy()
    combined["_recipe_key"] = recipe_key(combined)
    before_dedup = len(combined)
    # Later observed-history records supersede initial duplicates.
    combined = combined.drop_duplicates("_recipe_key", keep="last").reset_index(drop=True)

    if cfg.property_success_only and "success" in combined.columns:
        selected = combined[
            pd.to_numeric(combined["success"], errors="coerce") >= cfg.success_threshold
        ].reset_index(drop=True)
    else:
        selected = combined.reset_index(drop=True)

    if len(selected) < 30:
        raise ValueError(f"Too few property-regression rows after filtering: {len(selected)}")

    manifest = {
        "initial_rows_raw": int(len(initial)),
        "history_rows_with_targets": int(len(history)),
        "combined_rows_before_dedup": int(before_dedup),
        "combined_rows_after_dedup": int(len(combined)),
        "property_rows_used": int(len(selected)),
        "success_only": bool(cfg.property_success_only),
        "source_counts_used": selected["data_source"].value_counts().to_dict()
        if "data_source" in selected.columns else {},
    }
    return selected, manifest


class IndexedFPDataset(Dataset):
    """Precomputed FPNN inputs; y is kept in physical units."""
    def __init__(self, df: pd.DataFrame, featurizer):
        self.df = df.reset_index(drop=True)
        precursors = sorted(
            set(self.df["precursor_1"].astype(str))
            | set(self.df["precursor_2"].astype(str))
        )
        fp_map = {
            p: torch.tensor(featurizer.fingerprints(p), dtype=torch.float32)
            for p in precursors
        }
        reagents = sorted(set(self.df["reagent_1"].astype(str)))
        solv_map = {r: featurizer.solvent_feat(r).float() for r in reagents}

        fps1, fps2, globals_, ys = [], [], [], []
        for _, r in self.df.iterrows():
            p1, p2 = str(r["precursor_1"]), str(r["precursor_2"])
            s = solv_map[str(r["reagent_1"])]
            reaction = featurizer.reaction_feat(r["temperature"], r["time_h"]).float()
            pre1, pre2 = float(r["pre_1_mmol"]), float(r["pre_2_mmol"])
            ratio = pre1 / (pre1 + pre2) if (pre1 + pre2) > 0 else 0.5
            fps1.append(fp_map[p1])
            fps2.append(fp_map[p2])
            globals_.append(torch.cat([s, reaction, torch.tensor([ratio])]))
            ys.append(torch.tensor([r[t] for t in TARGETS], dtype=torch.float32))

        self.fps1 = torch.stack(fps1)
        self.fps2 = torch.stack(fps2)
        self.global_feat = torch.stack(globals_)
        self.y = torch.stack(ys)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.fps1[idx], self.fps2[idx], self.global_feat[idx], self.y[idx]


class SubsetDataset(Dataset):
    def __init__(self, base: IndexedFPDataset, indices):
        self.base = base
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[int(self.indices[idx])]


class FingerprintEncoderSmall(nn.Module):
    def __init__(self, input_dim=679, latent_dim=32, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, latent_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.float())


def make_process_encoder(name: str, input_dim: int):
    """Return encoder module and output dimension."""
    if name == "identity14":
        return nn.Identity(), input_dim
    if name == "linear8":
        return nn.Sequential(nn.Linear(input_dim, 8), nn.ReLU(), nn.Dropout(0.10)), 8
    if name == "linear16":
        return nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Dropout(0.10)), 16
    if name == "linear32":
        return nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.10)), 32
    if name == "two_layer_32_16":
        return nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(32, 16), nn.ReLU(),
        ), 16
    if name == "two_layer_64_16":
        return nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(64, 16), nn.ReLU(),
        ), 16
    if name == "two_layer_128_32":
        return nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 32), nn.ReLU(),
        ), 32
    if name == "old_two_layer_256_256":
        return nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, 256), nn.ReLU(),
        ), 256
    raise ValueError(f"Unknown process encoder: {name}")


class FPProcessNNAblation(nn.Module):
    def __init__(self, fp_dim: int, process_dim: int, encoder_name: str):
        super().__init__()
        self.fp_encoder = FingerprintEncoderSmall(input_dim=fp_dim)
        self.proc_encoder, proc_out = make_process_encoder(encoder_name, process_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + proc_out, 64), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(64, 3),
        )

    def forward(self, fps1, fps2, global_feat):
        f1 = self.fp_encoder(fps1)
        f2 = self.fp_encoder(fps2)
        pair = torch.cat([f1 + f2, torch.abs(f1 - f2)], dim=-1)
        fusion = self.fusion_mlp(pair)
        proc = self.proc_encoder(global_feat.float())
        return self.head(torch.cat([fusion, proc], dim=-1))


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_loader(dataset, indices, batch_size, shuffle):
    return DataLoader(
        SubsetDataset(dataset, indices),
        batch_size=min(batch_size, max(1, len(indices))),
        shuffle=shuffle,
        num_workers=0,
    )


def predict(model, loader, scaler, device):
    model.eval()
    pred_all, y_all = [], []
    with torch.no_grad():
        for fp1, fp2, proc, y in loader:
            pred_z = model(fp1.to(device), fp2.to(device), proc.to(device))
            pred_all.append(pred_z.cpu().numpy())
            y_all.append(y.numpy())
    pred_z = np.concatenate(pred_all)
    y_true = np.concatenate(y_all)
    y_pred = scaler.inverse_transform(pred_z)
    return y_true, y_pred


def metrics_rows(y_true, y_pred, split_name, metadata):
    rows = []
    for j, target in enumerate(TARGETS):
        rows.append({
            **metadata,
            "split": split_name,
            "target": target,
            "r2": r2_score(y_true[:, j], y_pred[:, j]),
            "mae": mean_absolute_error(y_true[:, j], y_pred[:, j]),
            "rmse": math.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j])),
            "n": len(y_true),
        })
    rows.append({
        **metadata,
        "split": split_name,
        "target": "MacroMean",
        "r2": np.mean([
            r2_score(y_true[:, j], y_pred[:, j]) for j in range(len(TARGETS))
        ]),
        "mae": np.mean([
            mean_absolute_error(y_true[:, j], y_pred[:, j]) for j in range(len(TARGETS))
        ]),
        "rmse": np.mean([
            math.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j]))
            for j in range(len(TARGETS))
        ]),
        "n": len(y_true),
    })
    return rows


def train_and_evaluate(
    dataset: IndexedFPDataset,
    outer_train_idx,
    outer_test_idx,
    encoder_name: str,
    cfg: Config,
    seed: int,
    train_fraction: float = 1.0,
):
    set_seed(seed)
    outer_train_idx = np.asarray(outer_train_idx, dtype=int)
    outer_test_idx = np.asarray(outer_test_idx, dtype=int)

    # Subsample only the outer training partition.
    if train_fraction < 1.0:
        rng = np.random.default_rng(seed)
        n_keep = max(12, int(round(len(outer_train_idx) * train_fraction)))
        n_keep = min(n_keep, len(outer_train_idx))
        used_train_idx = np.sort(rng.choice(outer_train_idx, size=n_keep, replace=False))
    else:
        used_train_idx = outer_train_idx.copy()

    fit_idx, val_idx = train_test_split(
        used_train_idx,
        test_size=cfg.inner_val_fraction,
        random_state=seed,
    )

    # Target scaler is fitted only on the inner fitting partition.
    scaler = StandardScaler().fit(dataset.y[fit_idx].numpy())

    train_loader = make_loader(dataset, fit_idx, cfg.batch_size, True)
    val_loader = make_loader(dataset, val_idx, cfg.batch_size, False)
    used_train_loader = make_loader(dataset, used_train_idx, cfg.batch_size, False)
    test_loader = make_loader(dataset, outer_test_idx, cfg.batch_size, False)

    fp_dim = dataset.fps1.shape[1]
    proc_dim = dataset.global_feat.shape[1]
    model = FPProcessNNAblation(fp_dim, proc_dim, encoder_name).to(cfg.device)
    n_parameters = count_parameters(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = np.inf
    wait = 0
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for fp1, fp2, proc, y in train_loader:
            y_z = torch.tensor(
                scaler.transform(y.numpy()).astype(np.float32),
                device=cfg.device,
            )
            optimizer.zero_grad()
            pred = model(fp1.to(cfg.device), fp2.to(cfg.device), proc.to(cfg.device))
            loss = loss_fn(pred, y_z)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for fp1, fp2, proc, y in val_loader:
                y_z = torch.tensor(
                    scaler.transform(y.numpy()).astype(np.float32),
                    device=cfg.device,
                )
                pred = model(fp1.to(cfg.device), fp2.to(cfg.device), proc.to(cfg.device))
                val_losses.append(loss_fn(pred, y_z).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
        if wait >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_train, p_train = predict(model, used_train_loader, scaler, cfg.device)
    y_test, p_test = predict(model, test_loader, scaler, cfg.device)

    metadata = {
        "encoder": encoder_name,
        "train_fraction": float(train_fraction),
        "n_outer_train_available": int(len(outer_train_idx)),
        "n_train_used": int(len(used_train_idx)),
        "n_fit": int(len(fit_idx)),
        "n_inner_val": int(len(val_idx)),
        "n_outer_test": int(len(outer_test_idx)),
        "best_epoch": int(best_epoch),
        "best_inner_val_loss": float(best_val),
        "n_parameters": n_parameters,
        "seed": int(seed),
    }
    rows = metrics_rows(y_train, p_train, "train_used", metadata)
    rows += metrics_rows(y_test, p_test, "outer_test", metadata)
    return rows


def summarize_fold_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        df.groupby(group_cols + ["split", "target"], as_index=False)
        .agg(
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            n_folds=("fold_id", "nunique"),
            n_parameters=("n_parameters", "first"),
            n_train_mean=("n_train_used", "mean"),
            n_test_mean=("n_outer_test", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
        )
    )

    # Add train-test generalization gap for R2.
    train = summary[summary["split"] == "train_used"][
        group_cols + ["target", "r2_mean"]
    ].rename(columns={"r2_mean": "train_r2_mean"})
    test = summary[summary["split"] == "outer_test"][
        group_cols + ["target", "r2_mean"]
    ].rename(columns={"r2_mean": "test_r2_mean"})
    gap = train.merge(test, on=group_cols + ["target"], how="inner")
    gap["generalization_gap_r2"] = gap["train_r2_mean"] - gap["test_r2_mean"]
    summary = summary.merge(
        gap[group_cols + ["target", "generalization_gap_r2"]],
        on=group_cols + ["target"],
        how="left",
    )
    return summary


def plot_encoder_ablation(summary: pd.DataFrame, out: Path):
    sub = summary[
        (summary["split"] == "outer_test")
        & (summary["target"] == "MacroMean")
    ].copy()
    order = list(sub.sort_values("r2_mean", ascending=False)["encoder"])
    sub["encoder"] = pd.Categorical(sub["encoder"], order, ordered=True)
    sub = sub.sort_values("encoder")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(sub))
    ax.bar(
        x, sub["r2_mean"], yerr=sub["r2_std"],
        color="#9DB7D5", edgecolor="#294C6B", linewidth=0.9,
        capsize=3,
    )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["encoder"], rotation=30, ha="right")
    ax.set_ylabel(r"Outer-CV macro-mean $R^2$")
    ax.set_title("Process-encoder width and depth ablation")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "Fig_process_encoder_ablation.png", dpi=600)
    fig.savefig(out / "Fig_process_encoder_ablation.pdf")
    plt.close(fig)


def plot_learning_curve(summary: pd.DataFrame, out: Path):
    test = summary[
        (summary["split"] == "outer_test")
        & (summary["target"] == "MacroMean")
    ].sort_values("train_fraction")
    train = summary[
        (summary["split"] == "train_used")
        & (summary["target"] == "MacroMean")
    ].sort_values("train_fraction")

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for sub, label, color, marker in [
        (train, "Training subset", "#D55E00", "s"),
        (test, "Fixed outer test folds", "#0072B2", "o"),
    ]:
        x = sub["n_train_mean"].to_numpy()
        y = sub["r2_mean"].to_numpy()
        sd = sub["r2_std"].fillna(0).to_numpy()
        ax.plot(x, y, marker=marker, lw=2, color=color, label=label)
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.16)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Number of samples used from each outer training fold")
    ax.set_ylabel(r"Macro-mean $R^2$")
    ax.set_title("Fixed-test learning curve")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "Fig_fixed_test_learning_curve.png", dpi=600)
    fig.savefig(out / "Fig_fixed_test_learning_curve.pdf")
    plt.close(fig)


def plot_combined(ablation_summary, learning_summary, out: Path):
    ab = ablation_summary[
        (ablation_summary["split"] == "outer_test")
        & (ablation_summary["target"] == "MacroMean")
    ].sort_values("r2_mean", ascending=False)
    te = learning_summary[
        (learning_summary["split"] == "outer_test")
        & (learning_summary["target"] == "MacroMean")
    ].sort_values("train_fraction")
    tr = learning_summary[
        (learning_summary["split"] == "train_used")
        & (learning_summary["target"] == "MacroMean")
    ].sort_values("train_fraction")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    x = np.arange(len(ab))
    axes[0].bar(
        x, ab["r2_mean"], yerr=ab["r2_std"],
        color="#B8CCE4", edgecolor="#2F5577", capsize=3,
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ab["encoder"], rotation=32, ha="right")
    axes[0].set_ylabel(r"Outer-CV macro-mean $R^2$")
    axes[0].set_title("(a) Process-encoder ablation")
    axes[0].axhline(0, color="black", lw=0.8)

    for sub, label, color, marker in [
        (tr, "Training subset", "#D55E00", "s"),
        (te, "Fixed outer test folds", "#0072B2", "o"),
    ]:
        xx = sub["n_train_mean"].to_numpy()
        yy = sub["r2_mean"].to_numpy()
        ss = sub["r2_std"].fillna(0).to_numpy()
        axes[1].plot(xx, yy, marker=marker, lw=2, color=color, label=label)
        axes[1].fill_between(xx, yy - ss, yy + ss, color=color, alpha=0.16)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("Training samples per outer fold")
    axes[1].set_ylabel(r"Macro-mean $R^2$")
    axes[1].set_title("(b) Fixed-test learning curve")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "Fig_ablation_and_learning_curve.png", dpi=600)
    fig.savefig(out / "Fig_ablation_and_learning_curve.pdf")
    plt.close(fig)


def main(cfg=None):
    cfg = cfg or Config()
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.cv_seed)

    ns = load_production_namespace(cfg.production_script)
    df, manifest = load_combined_property_data(cfg, ns)

    smiles = pd.read_excel(cfg.smiles_file)
    featurizer = ns["CarbonDotFeaturizer"](
        smiles, ns["get_solvent_polarity_data"]()
    )
    dataset = IndexedFPDataset(df, featurizer)
    manifest.update({
        "fingerprint_input_dim": int(dataset.fps1.shape[1]),
        "process_input_dim": int(dataset.global_feat.shape[1]),
        "targets": TARGETS,
        "config": asdict(cfg),
    })
    with open(out / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    rkf = RepeatedKFold(
        n_splits=cfg.n_splits,
        n_repeats=cfg.n_repeats,
        random_state=cfg.cv_seed,
    )
    splits = list(rkf.split(np.arange(len(dataset))))

    # A. Process-encoder ablation at 100% of each outer training fold.
    ablation_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        repeat_id = (fold_id - 1) // cfg.n_splits + 1
        fold_within_repeat = (fold_id - 1) % cfg.n_splits + 1
        for encoder_no, encoder_name in enumerate(cfg.encoder_names):
            seed = cfg.cv_seed + fold_id * 100 + encoder_no
            print(
                f"[Ablation] fold={fold_id}/{len(splits)}, "
                f"encoder={encoder_name}"
            )
            rows = train_and_evaluate(
                dataset, train_idx, test_idx, encoder_name, cfg, seed, 1.0
            )
            for row in rows:
                row.update({
                    "fold_id": fold_id,
                    "repeat_id": repeat_id,
                    "fold_within_repeat": fold_within_repeat,
                })
            ablation_rows.extend(rows)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(
        out / "process_encoder_fold_metrics.csv",
        index=False, encoding="utf-8-sig",
    )
    ablation_summary = summarize_fold_metrics(ablation_df, ["encoder"])
    ablation_summary.to_csv(
        out / "process_encoder_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    # B. Learning curve with a fixed outer test fold.
    curve_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
        repeat_id = (fold_id - 1) // cfg.n_splits + 1
        fold_within_repeat = (fold_id - 1) % cfg.n_splits + 1
        for frac_no, fraction in enumerate(cfg.train_fractions):
            seed = cfg.cv_seed + fold_id * 1000 + frac_no
            print(
                f"[Learning curve] fold={fold_id}/{len(splits)}, "
                f"fraction={fraction:.2f}"
            )
            rows = train_and_evaluate(
                dataset, train_idx, test_idx,
                cfg.learning_curve_encoder, cfg, seed, fraction,
            )
            for row in rows:
                row.update({
                    "fold_id": fold_id,
                    "repeat_id": repeat_id,
                    "fold_within_repeat": fold_within_repeat,
                })
            curve_rows.extend(rows)

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(
        out / "learning_curve_fold_metrics.csv",
        index=False, encoding="utf-8-sig",
    )
    curve_summary = summarize_fold_metrics(
        curve_df, ["encoder", "train_fraction"]
    )
    curve_summary.to_csv(
        out / "learning_curve_summary.csv",
        index=False, encoding="utf-8-sig",
    )

    plot_encoder_ablation(ablation_summary, out)
    plot_learning_curve(curve_summary, out)
    plot_combined(ablation_summary, curve_summary, out)

    print("\nProcess-encoder macro-mean outer-CV summary:")
    print(
        ablation_summary[
            (ablation_summary["split"] == "outer_test")
            & (ablation_summary["target"] == "MacroMean")
        ][
            ["encoder", "r2_mean", "r2_std", "n_parameters",
             "generalization_gap_r2"]
        ].sort_values("r2_mean", ascending=False).to_string(index=False)
    )
    print(f"\nAll results saved to: {out.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate process encoders and learning curves.')
    parser.add_argument("--history", type=Path, help="Include a local recipe-level history in this retrospective analysis.")
    parser.add_argument("--quick", action="store_true", help="Exercise the analysis with two folds and a reduced fit budget.")
    args = parser.parse_args()
    cfg = Config()
    if args.history:
        if not args.history.is_file():
            parser.error(f"History file does not exist: {args.history}")
        cfg.history_file = str(args.history)
    if args.quick:
        cfg.n_splits = 2
        cfg.n_repeats = 1
        cfg.epochs = 1
        cfg.train_fractions = (0.5, 1.0)
        cfg.output_dir = "outputs/quick/process_encoder"
    main(cfg)
