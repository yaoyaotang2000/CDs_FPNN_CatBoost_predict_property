# fp_process_catboost_ensemble_cv.py
# -*- coding: utf-8 -*-
"""
FP+Process NN + CatBoost integrated training and ensemble prediction.

Revision for success-labelled data
-----------------------------------
- Optical-property comparison uses only rows with success >= 0.5.
- success=0 rows are never used by the Ex/Em/QY regressors.
- CatBoost features are rebuilt from the same filtered rows using the updated
  02_train_fp_catboost_cycle.py feature-engineering functions.
- The outer test fold is not used for NN or CatBoost early stopping; an inner
  validation subset is drawn only from the outer training fold.

This script integrates:

1. FP+Process neural network
   - Precursor fingerprints + process/global features.
   - Multi-output prediction for Ex_max_nm, Em_max_nm, QY_%.
   - y z-score scaling, consistent with previous NN code.

2. CatBoost single-task models
   - One CatBoostRegressor for each target.
   - Uses selected feature files, consistent with the uploaded standalone CatBoost code.

3. Ensemble predictions
   - Simple average and fixed-weight averages.
   - Outputs fold-level metrics, summary metrics, OOF predictions, and figures.

Run:
    python fp_process_catboost_ensemble_cv.py

Default required files:
    ./data/output_with_molar_amount_pubchem.xlsx
    ./data/compound_smiles.xlsx
    ./model/model.py

Default CatBoost selected-feature files:
    ./cd_catboost_feature_selection/Ex_max_nm_best_top20_selected_features.csv
    ./cd_catboost_feature_selection/Em_max_nm_best_top20_selected_features.csv
    ./cd_catboost_feature_selection/QY_%_best_top20_selected_features.csv

If your CatBoost selected-feature file names are different, modify CATBOOST_FEATURE_FILES.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple
import importlib.util
import sys

import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from torch_geometric.data import Batch

from model.model import data_process, CarbonDotFeaturizer


TARGET_NAMES = ["Ex_max_nm", "Em_max_nm", "QY_%"]


CATBOOST_FEATURE_FILES = {
    "Ex_max_nm": "cd_catboost_feature_selection/Ex_max_nm_best_top20_feature_list.csv",
    "Em_max_nm": "cd_catboost_feature_selection/Em_max_nm_best_top35_feature_list.csv",
    "QY_%": "cd_catboost_feature_selection/QY_%_best_top35_feature_list.csv",
}

META_LIKE_COLS = [
    "precursor_1", "precursor_2", "reagent_1",
    "Ex_max_nm", "Em_max_nm", "QY_%", "success",
]

FORBIDDEN_FEATURES = {
    "Ex_max_nm", "Em_max_nm", "QY_%", "success",
    "stoke_shift", "stoke_shift_calc_check", "Stokes_shift_nm",
}


@dataclass
class DataCfg:
    data_path: str = "./data/output_with_molar_amount_pubchem.xlsx"
    smiles_path: str = "./data/compound_smiles.xlsx"
    training_script_path: str = "./validation/02_train_fp_catboost_cycle_success_selected_features.py"
    success_column: str = "success"
    success_positive_threshold: float = 0.5
    success_only: bool = True
    minimum_success_rows: int = 20
    smiles_compound_sheet: str = "compound"


@dataclass
class TrainCfg:
    epochs: int = 200
    lr: float = 2e-3
    weight_decay: float = 1e-4
    batch_size: int = 10
    patience: int = 20
    grad_clip: float = 1.0


@dataclass
class CVCfg:
    seed: int = 42
    n_splits: int = 5
    n_repeats: int = 3
    num_workers: int = 0
    inner_val_fraction: float = 0.15


@dataclass
class OutputCfg:
    save_dir: str = "./fp_process_catboost_ensemble_success_only_results"
    dpi: int = 300


@dataclass
class CatBoostCfg:
    iterations: int = 2000
    learning_rate: float = 0.05
    depth: int = 4
    l2_leaf_reg: float = 5.0
    random_strength: float = 1.0
    subsample: float = 0.9
    early_stopping_rounds: int = 100


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def count_trainable_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def batch_to_device(batch: Dict[str, Any], dev: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(dev)
        else:
            try:
                out[k] = v.to(dev)
            except Exception:
                out[k] = v
    return out


def calc_metrics_multi(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m: Dict[str, float] = {}
    for j, name in enumerate(TARGET_NAMES):
        m[f"{name}_MAE"] = float(mean_absolute_error(y_true[:, j], y_pred[:, j]))
        m[f"{name}_RMSE"] = rmse(y_true[:, j], y_pred[:, j])
        m[f"{name}_R2"] = float(r2_score(y_true[:, j], y_pred[:, j]))

    m["MAE_mean"] = float(np.mean([m[f"{t}_MAE"] for t in TARGET_NAMES]))
    m["RMSE_mean"] = float(np.mean([m[f"{t}_RMSE"] for t in TARGET_NAMES]))
    m["R2_mean"] = float(np.mean([m[f"{t}_R2"] for t in TARGET_NAMES]))
    return m


def get_y_raw(dataset: Dataset, indices: List[int]) -> np.ndarray:
    ys = []
    for i in indices:
        y = dataset[i]["y"]
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        ys.append(np.asarray(y, dtype=np.float32).reshape(1, -1))
    return np.vstack(ys).astype(np.float32)


def summarize_metrics(fold_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c for c in fold_df.columns
        if c.endswith("_MAE") or c.endswith("_RMSE") or c.endswith("_R2")
        or c in ["MAE_mean", "RMSE_mean", "R2_mean"]
    ]

    rows = []
    for (model, split), g in fold_df.groupby(["model", "split"]):
        row = {"model": model, "split": split}
        for c in metric_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"{c}_mean"] = float(vals.mean())
            row[f"{c}_std"] = float(vals.std(ddof=1))
        rows.append(row)

    return pd.DataFrame(rows)


def get_solvent_polarity_data() -> Dict[str, Dict[str, float]]:
    return {
        "water": {"dielectric_constant": 80.1, "pi_star": 1.09, "HBD": 1.17, "HBA": 0.47, "ET30": 63.1, "pKa": 15.7, "ORP_mV": 300, "density_g_cm3": 0.997, "specific_heat_J_gK": 4.18, "viscosity_mPa_s": 0.89, "boiling_point_C": 100},
        "dimethylformamide": {"dielectric_constant": 36.7, "pi_star": 0.88, "HBD": 0.00, "HBA": 0.69, "ET30": 43.2, "pKa": 29, "ORP_mV": 200, "density_g_cm3": 0.944, "specific_heat_J_gK": 2.47, "viscosity_mPa_s": 0.92, "boiling_point_C": 153},
        "ethanol": {"dielectric_constant": 24.6, "pi_star": 0.54, "HBD": 0.86, "HBA": 0.75, "ET30": 51.9, "pKa": 15.9, "ORP_mV": 250, "density_g_cm3": 0.789, "specific_heat_J_gK": 2.44, "viscosity_mPa_s": 1.07, "boiling_point_C": 78.37},
        "sulfuric acid": {"dielectric_constant": 83, "pi_star": 1.11, "HBD": 1.22, "HBA": 0.48, "ET30": 64.8, "pKa": 0.0, "ORP_mV": 800, "density_g_cm3": 1.05, "specific_heat_J_gK": 3.6, "viscosity_mPa_s": 1.2, "boiling_point_C": 100},
        "phosphoric acid": {"dielectric_constant": 82, "pi_star": 1.10, "HBD": 1.20, "HBA": 0.47, "ET30": 64.0, "pKa": 1.0, "ORP_mV": 700, "density_g_cm3": 1.03, "specific_heat_J_gK": 3.8, "viscosity_mPa_s": 1.1, "boiling_point_C": 100},
    }


class CarbonDotDataset(Dataset):
    def __init__(self, df, featurizer):
        self.df = df.reset_index(drop=True)
        self.featurizer = featurizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        g1 = self.featurizer.smiles_to_graph(row["precursor_1"])
        g2 = self.featurizer.smiles_to_graph(row["precursor_2"])

        fp1 = torch.tensor(self.featurizer.fingerprints(row["precursor_1"]), dtype=torch.float)
        fp2 = torch.tensor(self.featurizer.fingerprints(row["precursor_2"]), dtype=torch.float)

        d1 = self.featurizer.descriptors(row["precursor_1"])
        d2 = self.featurizer.descriptors(row["precursor_2"])

        s_feat = self.featurizer.solvent_feat(row["reagent_1"])
        r_feat = self.featurizer.reaction_feat(row["temperature"], row["time_h"])

        pre1 = float(row["pre_1_mmol"])
        pre2 = float(row["pre_2_mmol"])
        ratio = pre1 / (pre1 + pre2) if (pre1 + pre2) > 0 else 0.5
        pre_ratio = torch.tensor([ratio], dtype=torch.float)

        global_feat = torch.cat([s_feat, r_feat, pre_ratio], dim=0)

        y = torch.tensor(
            [row["Ex_max_nm"], row["Em_max_nm"], row["QY_%"]],
            dtype=torch.float,
        )

        return {
            "g1": g1,
            "g2": g2,
            "fps_1": fp1,
            "fps_2": fp2,
            "desc_1": d1,
            "desc_2": d2,
            "global_feat": global_feat,
            "y": y,
        }


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    batch_graphs_1 = Batch.from_data_list([item["g1"] for item in batch])
    batch_graphs_2 = Batch.from_data_list([item["g2"] for item in batch])

    return {
        "graphs_1": batch_graphs_1,
        "graphs_2": batch_graphs_2,
        "fps_1": torch.stack([item["fps_1"] for item in batch], dim=0),
        "fps_2": torch.stack([item["fps_2"] for item in batch], dim=0),
        "desc_1": torch.stack([item["desc_1"] for item in batch], dim=0),
        "desc_2": torch.stack([item["desc_2"] for item in batch], dim=0),
        "global_feat": torch.stack([item["global_feat"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
    }


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p.resolve()}")
    return pd.read_excel(p) if p.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(p)


def load_updated_02_module(path: str):
    """Load the exact updated 02 source, including its feature engineering."""
    p = Path(path).resolve()
    if not p.exists():
        # Also accept the uploaded/pasted filename used in this workspace.
        candidates = sorted(Path.cwd().glob("02*.py")) + sorted(Path.cwd().glob("*02*.txt"))
        candidates += sorted((Path.cwd() / "upload").glob("*2*.txt"))
        matches = []
        for c in candidates:
            text = c.read_text(encoding="utf-8-sig", errors="ignore")
            if "def build_dataset_features" in text and "def ensure_shift_and_success_columns" in text:
                matches.append(c)
        if not matches:
            raise FileNotFoundError(
                f"Updated 02 script not found: {p}. Set DataCfg.training_script_path."
            )
        p = matches[0].resolve()
    source = p.read_text(encoding="utf-8-sig")
    module_name = "updated_02_feature_engineering"
    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(p))
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(p)
    sys.modules[module_name] = module
    exec(compile(source, str(p), "exec"), module.__dict__)
    required = [
        "ensure_shift_and_success_columns", "build_compound_feature_table",
        "build_dataset_features", "read_feature_list",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"Updated 02 script is missing definitions: {missing}")
    print(f"Using updated 02 feature engineering: {p}")
    return module


def load_success_only_aligned_data(data_cfg: DataCfg):
    """Build aligned NN/CatBoost inputs from the same success=1 rows."""
    helper = load_updated_02_module(data_cfg.training_script_path)
    raw = read_table(data_cfg.data_path)
    raw = helper.ensure_shift_and_success_columns(raw)
    n_raw = len(raw)

    if data_cfg.success_column not in raw.columns:
        raise ValueError(
            f"Required column '{data_cfg.success_column}' is absent. "
            "Property-model comparison requires an explicit success label."
        )
    success_numeric = pd.to_numeric(raw[data_cfg.success_column], errors="coerce")
    n_missing_success = int(success_numeric.isna().sum())
    n_failure = int((success_numeric < data_cfg.success_positive_threshold).sum())
    n_success = int((success_numeric >= data_cfg.success_positive_threshold).sum())

    if data_cfg.success_only:
        raw = raw.loc[success_numeric >= data_cfg.success_positive_threshold].copy()
    if len(raw) < data_cfg.minimum_success_rows:
        raise ValueError(
            f"Only {len(raw)} success rows remain; minimum_success_rows="
            f"{data_cfg.minimum_success_rows}. No fallback to success=0 rows is allowed."
        )

    # Match the property-regression path in updated 02.
    processed = data_process(raw).reset_index(drop=True)
    processed = helper.ensure_shift_and_success_columns(processed)
    required = [
        "precursor_1", "precursor_2", "reagent_1", "temperature", "time_h",
        "pre_1_mmol", "pre_2_mmol", *TARGET_NAMES,
    ]
    missing = [c for c in required if c not in processed.columns]
    if missing:
        raise ValueError(f"Property data missing required columns: {missing}")
    processed = processed.dropna(subset=required).reset_index(drop=True)
    if data_cfg.success_only:
        final_success = pd.to_numeric(processed[data_cfg.success_column], errors="coerce")
        processed = processed.loc[
            final_success >= data_cfg.success_positive_threshold
        ].reset_index(drop=True)

    # NN and CatBoost now receive exactly the same rows and order.
    smiles_df = pd.read_excel(data_cfg.smiles_path)
    featurizer = CarbonDotFeaturizer(
        df_smiles=smiles_df,
        solvent_polarity_data=get_solvent_polarity_data(),
    )
    nn_dataset = CarbonDotDataset(processed, featurizer)

    try:
        smiles_compound = pd.read_excel(
            data_cfg.smiles_path, sheet_name=data_cfg.smiles_compound_sheet
        )
    except ValueError:
        smiles_compound = pd.read_excel(data_cfg.smiles_path)
    compound_feat_df, fp_objs = helper.build_compound_feature_table(smiles_compound)
    feature_df = helper.build_dataset_features(
        processed, compound_feat_df, fp_objs, has_targets=True
    ).reset_index(drop=True)
    if len(feature_df) != len(processed):
        raise ValueError(
            f"02 feature generation changed row count: property={len(processed)}, "
            f"features={len(feature_df)}"
        )

    cat_data = {}
    for target in TARGET_NAMES:
        feature_path = Path(CATBOOST_FEATURE_FILES[target])
        if not feature_path.exists():
            raise FileNotFoundError(f"Selected feature-list file not found: {feature_path}")
        selected = helper.read_feature_list(str(feature_path))
        leakage = [c for c in selected if c in FORBIDDEN_FEATURES]
        if leakage:
            raise ValueError(f"Forbidden/leakage features for {target}: {leakage}")
        missing_features = [c for c in selected if c not in feature_df.columns]
        if missing_features:
            raise ValueError(
                f"Updated 02 feature table is missing selected features for {target}: "
                f"{missing_features[:30]}"
            )
        X = feature_df[selected].copy()
        for c in selected:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.fillna(0.0)
        y = processed[target].astype(float).reset_index(drop=True)
        cat_data[target] = {
            "df": feature_df, "X": X.reset_index(drop=True),
            "y": y, "feature_cols": selected,
        }
        print(f"[CatBoost data] {target}: samples={len(y)}, features={len(selected)}")

    audit = pd.DataFrame([{
        "n_raw_rows": n_raw,
        "n_success_label_missing": n_missing_success,
        "n_success_0_removed": n_failure if data_cfg.success_only else 0,
        "n_success_1_before_property_cleaning": n_success,
        "n_final_property_rows": len(processed),
        "success_threshold": data_cfg.success_positive_threshold,
        "success_only": data_cfg.success_only,
        "data_path": str(Path(data_cfg.data_path).resolve()),
        "updated_02_path": str(Path(helper.__file__).resolve()),
    }])
    print("\n[SUCCESS FILTER AUDIT]")
    print(audit.to_string(index=False))
    return processed, nn_dataset, cat_data, audit


def validate_alignment(nn_df: pd.DataFrame, cat_data: Dict[str, Dict[str, Any]]) -> None:
    n = len(nn_df)
    for target in TARGET_NAMES:
        y_nn = nn_df[target].astype(float).values
        y_cb = cat_data[target]["y"].astype(float).values

        if len(y_cb) != n:
            raise ValueError(
                f"Sample size mismatch for {target}: NN={n}, CatBoost={len(y_cb)}. "
                "Please make sure the CatBoost feature files use the same rows/order as the NN dataset."
            )

        max_abs_diff = float(np.max(np.abs(y_nn - y_cb)))
        if max_abs_diff > 1e-5:
            print(
                f"Warning: target values for {target} differ between NN data and CatBoost data. "
                f"max_abs_diff={max_abs_diff:.6f}. Please verify row alignment."
            )


class FingerprintEncoderSmall(nn.Module):
    def __init__(self, input_dim: int = 679, latent_dim: int = 32, dropout: float = 0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class ProcessEncoderSmall(nn.Module):
    def __init__(self, input_dim: int = 14, out_dim: int = 16, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class FPProcessNN(nn.Module):
    def __init__(self):
        super().__init__()
        fp_latent = 32
        proc_out = 16
        hidden = 64

        self.fp_encoder = FingerprintEncoderSmall(input_dim=679, latent_dim=fp_latent, dropout=0.30)
        self.proc_encoder = ProcessEncoderSmall(input_dim=14, out_dim=proc_out, dropout=0.10)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(2 * fp_latent, hidden),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden + proc_out, hidden),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(hidden, 3),
        )

    def forward(self, graphs_1, graphs_2, fps_1, fps_2, desc_1, desc_2, global_feat):
        f1 = self.fp_encoder(fps_1)
        f2 = self.fp_encoder(fps_2)
        fusion = self.fusion_mlp(torch.cat([f1 + f2, torch.abs(f1 - f2)], dim=-1))
        proc = self.proc_encoder(global_feat)
        z = torch.cat([fusion, proc], dim=-1)
        return self.head(z)


def train_fp_process_nn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_scaler: StandardScaler,
    dev: torch.device,
    cfg: TrainCfg,
) -> nn.Module:
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            y_raw = batch["y"].numpy().astype(np.float32)
            y_scaled = y_scaler.transform(y_raw).astype(np.float32)
            batch["y_scaled"] = torch.as_tensor(y_scaled, dtype=torch.float32)
            batch = batch_to_device(batch, dev)

            pred = model(
                batch["graphs_1"], batch["graphs_2"],
                batch["fps_1"], batch["fps_2"],
                batch["desc_1"], batch["desc_2"],
                batch["global_feat"],
            )
            loss = F.mse_loss(pred, batch["y_scaled"], reduction="mean")

            opt.zero_grad()
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                y_raw = batch["y"].numpy().astype(np.float32)
                y_scaled = y_scaler.transform(y_raw).astype(np.float32)
                batch["y_scaled"] = torch.as_tensor(y_scaled, dtype=torch.float32)
                batch = batch_to_device(batch, dev)

                pred = model(
                    batch["graphs_1"], batch["graphs_2"],
                    batch["fps_1"], batch["fps_2"],
                    batch["desc_1"], batch["desc_2"],
                    batch["global_feat"],
                )
                val_losses.append(F.mse_loss(pred, batch["y_scaled"], reduction="mean").item())

        val_loss = float(np.mean(val_losses))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_nn(model: nn.Module, loader: DataLoader, y_scaler: StandardScaler, dev: torch.device) -> np.ndarray:
    model.eval()
    model.to(dev)
    preds_scaled = []

    for batch in loader:
        batch = batch_to_device(batch, dev)
        pred = model(
            batch["graphs_1"], batch["graphs_2"],
            batch["fps_1"], batch["fps_2"],
            batch["desc_1"], batch["desc_2"],
            batch["global_feat"],
        ).detach().cpu().numpy().astype(np.float32)
        preds_scaled.append(pred)

    preds_scaled = np.vstack(preds_scaled)
    return y_scaler.inverse_transform(preds_scaled).astype(np.float32)


def make_catboost_model(seed: int, cfg: CatBoostCfg) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=cfg.iterations,
        learning_rate=cfg.learning_rate,
        depth=cfg.depth,
        l2_leaf_reg=cfg.l2_leaf_reg,
        random_strength=cfg.random_strength,
        bootstrap_type="Bernoulli",
        subsample=cfg.subsample,
        grow_policy="SymmetricTree",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def train_catboost_for_fold(
    cat_data: Dict[str, Dict[str, Any]],
    fit_idx: List[int],
    val_idx: List[int],
    tr_idx: List[int],
    te_idx: List[int],
    run_id: int,
    seed: int,
    cfg: CatBoostCfg,
) -> Tuple[np.ndarray, np.ndarray]:
    pred_train = np.zeros((len(tr_idx), len(TARGET_NAMES)), dtype=np.float32)
    pred_test = np.zeros((len(te_idx), len(TARGET_NAMES)), dtype=np.float32)

    for j, target in enumerate(TARGET_NAMES):
        X = cat_data[target]["X"]
        y = cat_data[target]["y"]

        X_fit = X.iloc[fit_idx]
        X_val = X.iloc[val_idx]
        y_fit = y.iloc[fit_idx]
        y_val = y.iloc[val_idx]
        X_tr = X.iloc[tr_idx]
        X_te = X.iloc[te_idx]

        model = make_catboost_model(seed=seed + run_id * 10 + j, cfg=cfg)
        model.fit(
            X_fit, y_fit,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=cfg.early_stopping_rounds,
            verbose=False,
        )

        pred_train[:, j] = model.predict(X_tr)
        pred_test[:, j] = model.predict(X_te)

    return pred_train, pred_test


def make_ensemble_predictions(pred_nn: np.ndarray, pred_cb: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    out["Ensemble_mean_0p5NN_0p5CB"] = 0.5 * pred_nn + 0.5 * pred_cb
    out["Ensemble_weight_0p3CB"] = 0.7 * pred_nn + 0.3 * pred_cb
    out["Ensemble_weight_0p7CB"] = 0.3 * pred_nn + 0.7 * pred_cb
    target_weight_cb = np.array([0.5, 0.5, 0.6], dtype=np.float32).reshape(1, -1)
    out["Ensemble_target_weighted"] = (1.0 - target_weight_cb) * pred_nn + target_weight_cb * pred_cb
    return out


def run_integrated_cv(
    nn_dataset: CarbonDotDataset,
    cat_data: Dict[str, Dict[str, Any]],
    train_cfg: TrainCfg,
    cv_cfg: CVCfg,
    cb_cfg: CatBoostCfg,
    output_cfg: OutputCfg,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(cv_cfg.seed)
    dev = get_device()

    save_dir = Path(output_cfg.save_dir)
    data_dir = save_dir / "data"
    fig_dir = save_dir / "figures"
    ensure_dir(data_dir)
    ensure_dir(fig_dir)

    idx = np.arange(len(nn_dataset), dtype=int)
    rkf = RepeatedKFold(n_splits=cv_cfg.n_splits, n_repeats=cv_cfg.n_repeats, random_state=cv_cfg.seed)

    fold_records = []
    oof_records = []

    temp_nn = FPProcessNN()
    param_df = pd.DataFrame([{"model": "FPProcessNN", "n_params": count_trainable_params(temp_nn)}])
    param_df.to_csv(data_dir / "parameter_count.csv", index=False, encoding="utf-8-sig")

    print(f"Using device: {dev}")
    print(f"FPProcessNN params: {count_trainable_params(temp_nn)}")

    for run_id, (tr_rel, te_rel) in enumerate(rkf.split(idx), 1):
        set_seed(cv_cfg.seed + run_id)

        tr = idx[tr_rel].tolist()
        te = idx[te_rel].tolist()
        fit_idx, val_idx = train_test_split(
            tr, test_size=cv_cfg.inner_val_fraction,
            random_state=cv_cfg.seed + run_id, shuffle=True,
        )
        fit_idx, val_idx = list(fit_idx), list(val_idx)
        rep = (run_id - 1) // cv_cfg.n_splits + 1
        fold = (run_id - 1) % cv_cfg.n_splits + 1

        print("\n" + "=" * 80)
        print(f"[Rep {rep}/{cv_cfg.n_repeats} | Fold {fold}/{cv_cfg.n_splits}]")
        print("=" * 80)

        fit_set = Subset(nn_dataset, fit_idx)
        val_set = Subset(nn_dataset, val_idx)
        train_set = Subset(nn_dataset, tr)
        test_set = Subset(nn_dataset, te)

        fit_loader = DataLoader(
            fit_set, batch_size=train_cfg.batch_size, shuffle=True,
            collate_fn=custom_collate_fn, num_workers=cv_cfg.num_workers,
        )
        val_loader = DataLoader(
            val_set, batch_size=train_cfg.batch_size, shuffle=False,
            collate_fn=custom_collate_fn, num_workers=cv_cfg.num_workers,
        )
        train_loader = DataLoader(
            train_set, batch_size=train_cfg.batch_size, shuffle=False,
            collate_fn=custom_collate_fn, num_workers=cv_cfg.num_workers,
        )
        test_loader = DataLoader(
            test_set, batch_size=train_cfg.batch_size, shuffle=False,
            collate_fn=custom_collate_fn, num_workers=cv_cfg.num_workers,
        )

        y_fit_raw = get_y_raw(nn_dataset, fit_idx)
        y_tr_raw = get_y_raw(nn_dataset, tr)
        y_te_raw = get_y_raw(nn_dataset, te)
        y_scaler = StandardScaler().fit(y_fit_raw)

        nn_model = FPProcessNN()
        nn_model = train_fp_process_nn(
            model=nn_model,
            train_loader=fit_loader,
            val_loader=val_loader,
            y_scaler=y_scaler,
            dev=dev,
            cfg=train_cfg,
        )

        pred_nn_train = predict_nn(nn_model, train_loader, y_scaler, dev)
        pred_nn_test = predict_nn(nn_model, test_loader, y_scaler, dev)

        pred_cb_train, pred_cb_test = train_catboost_for_fold(
            cat_data=cat_data, fit_idx=fit_idx, val_idx=val_idx,
            tr_idx=tr, te_idx=te,
            run_id=run_id, seed=cv_cfg.seed, cfg=cb_cfg,
        )

        train_preds = {"FPProcessNN": pred_nn_train, "CatBoost": pred_cb_train}
        test_preds = {"FPProcessNN": pred_nn_test, "CatBoost": pred_cb_test}

        train_preds.update(make_ensemble_predictions(pred_nn_train, pred_cb_train))
        test_preds.update(make_ensemble_predictions(pred_nn_test, pred_cb_test))

        for model_name, pred_train in train_preds.items():
            met = calc_metrics_multi(y_tr_raw, pred_train)
            fold_records.append({
                "model": model_name, "split": "train", "repeat": rep, "fold": fold,
                "run_id": run_id, "n_fit": len(fit_idx), "n_val": len(val_idx),
                "n_train": len(tr), "n_test": len(te), **met,
            })

        for model_name, pred_test in test_preds.items():
            met = calc_metrics_multi(y_te_raw, pred_test)
            fold_records.append({
                "model": model_name, "split": "test", "repeat": rep, "fold": fold,
                "run_id": run_id, "n_fit": len(fit_idx), "n_val": len(val_idx),
                "n_train": len(tr), "n_test": len(te), **met,
            })
            print(
                f"{model_name:28s} | "
                f"Test R2_mean={met['R2_mean']:.3f} | "
                f"Ex={met['Ex_max_nm_R2']:.3f} | "
                f"Em={met['Em_max_nm_R2']:.3f} | "
                f"QY={met['QY_%_R2']:.3f}"
            )

        for local_i, global_i in enumerate(te):
            rec = {
                "sample_index": global_i,
                "repeat": rep,
                "fold": fold,
                "run_id": run_id,
                "y_true_Ex_max_nm": y_te_raw[local_i, 0],
                "y_true_Em_max_nm": y_te_raw[local_i, 1],
                "y_true_QY_%": y_te_raw[local_i, 2],
            }
            for model_name, pred_test in test_preds.items():
                rec[f"{model_name}_pred_Ex_max_nm"] = pred_test[local_i, 0]
                rec[f"{model_name}_pred_Em_max_nm"] = pred_test[local_i, 1]
                rec[f"{model_name}_pred_QY_%"] = pred_test[local_i, 2]
            oof_records.append(rec)

    fold_df = pd.DataFrame(fold_records)
    summary_df = summarize_metrics(fold_df)
    oof_df = pd.DataFrame(oof_records)

    fold_df.to_csv(data_dir / "all_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(data_dir / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    oof_df.to_csv(data_dir / "all_oof_predictions.csv", index=False, encoding="utf-8-sig")

    plot_all(summary_df, fig_dir, output_cfg)

    print("\nSaved:")
    print(data_dir / "all_fold_metrics.csv")
    print(data_dir / "summary_metrics.csv")
    print(data_dir / "all_oof_predictions.csv")
    print(fig_dir)

    return fold_df, summary_df, oof_df


def get_test_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df[summary_df["split"] == "test"].copy()


def plot_bar_metric(summary_df, metric_mean_col, metric_std_col, ylabel, title, fname, fig_dir, output_cfg, ascending=False):
    df = get_test_summary(summary_df).sort_values(metric_mean_col, ascending=ascending)
    plt.figure(figsize=(9.0, 4.8))
    plt.bar(df["model"], df[metric_mean_col], yerr=df[metric_std_col], capsize=4)
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{fname}.png", dpi=output_cfg.dpi)
    plt.savefig(fig_dir / f"{fname}.svg")
    plt.close()


def plot_target_r2(summary_df, fig_dir, output_cfg):
    df = get_test_summary(summary_df).sort_values("R2_mean_mean", ascending=False)
    models = df["model"].tolist()
    targets = ["Ex_max_nm_R2", "Em_max_nm_R2", "QY_%_R2"]
    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10.0, 5.0))
    for j, target in enumerate(targets):
        plt.bar(
            x + (j - 1) * width,
            df[f"{target}_mean"].values,
            width=width,
            yerr=df[f"{target}_std"].values,
            capsize=3,
            label=target.replace("_R2", ""),
        )
    plt.ylabel("Test R²")
    plt.xlabel("Model")
    plt.title("Target-wise test R² comparison")
    plt.xticks(x, models, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_target_R2_comparison.png", dpi=output_cfg.dpi)
    plt.savefig(fig_dir / "fig2_target_R2_comparison.svg")
    plt.close()


def plot_all(summary_df, fig_dir, output_cfg):
    plot_bar_metric(
        summary_df, "R2_mean_mean", "R2_mean_std",
        "Mean test R²", "FP+Process NN, CatBoost, and ensemble comparison",
        "fig1_test_R2_mean", fig_dir, output_cfg, ascending=False,
    )
    plot_target_r2(summary_df, fig_dir, output_cfg)
    plot_bar_metric(
        summary_df, "MAE_mean_mean", "MAE_mean_std",
        "Mean test MAE", "Mean test MAE comparison",
        "fig3_test_MAE_mean", fig_dir, output_cfg, ascending=True,
    )
    plot_bar_metric(
        summary_df, "QY_%_R2_mean", "QY_%_R2_std",
        "QY test R²", "QY prediction comparison",
        "fig4_qy_R2_comparison", fig_dir, output_cfg, ascending=False,
    )


def main() -> None:
    set_seed(42)

    data_cfg = DataCfg()
    train_cfg = TrainCfg()
    cv_cfg = CVCfg()
    cb_cfg = CatBoostCfg()
    output_cfg = OutputCfg()

    print("Loading and filtering success=1 property data...")
    nn_df, nn_dataset, cat_data, audit = load_success_only_aligned_data(data_cfg)
    print(f"Aligned success-only dataset: samples={len(nn_dataset)}")

    audit_dir = Path(output_cfg.save_dir) / "data"
    ensure_dir(audit_dir)
    audit.to_csv(audit_dir / "success_filter_audit.csv", index=False, encoding="utf-8-sig")
    nn_df.to_csv(audit_dir / "success_only_property_rows.csv", index=False, encoding="utf-8-sig")

    print("Checking alignment...")
    validate_alignment(nn_df, cat_data)

    print("\nStarting integrated CV...")
    fold_df, summary_df, oof_df = run_integrated_cv(
        nn_dataset=nn_dataset,
        cat_data=cat_data,
        train_cfg=train_cfg,
        cv_cfg=cv_cfg,
        cb_cfg=cb_cfg,
        output_cfg=output_cfg,
    )

    print("\n====================")
    print("Summary metrics")
    print("====================")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
