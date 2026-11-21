# -*- coding: utf-8 -*-
"""
Paired retrospective BO benchmark using the exact model employed in the
closed-loop carbon-dot workflow:

    Ensemble_mean_0p5NN_0p5CB
    = mean of 5 FPProcessNN members + 5 target-specific CatBoost members

Because both model families contain the same number of members, concatenating
all members and taking their mean is exactly equivalent to:

    0.5 * mean(FPProcessNN members) + 0.5 * mean(CatBoost members)

The ensemble standard deviation is calculated from all 10 member predictions,
matching the BO prediction script. The color-specific UCB objective also follows
the original BO implementation:

    score_mu = ColorFit(Em) * [alpha * QY_score + (1-alpha) * Stokes_score]
    score_sigma = ColorFit(Em) * [alpha * QY_std_norm
                                  + (1-alpha) * Stokes_std_norm]
    UCB = score_mu + beta * score_sigma

ANTI-LEAKAGE RULE
-----------------
For every retrospective repeat and every selection step, the FPProcessNN and
CatBoost models are retrained ONLY on the currently labeled historical rows.
The unselected rows are treated as an unseen candidate pool. Do not replace
this with models trained once on all historical data.

MODEL CONSISTENCY
-----------------
The script locally defines the exact FPProcessNN architecture and training
procedure. It imports only the CatBoost feature-engineering functions from the
same Step-2 training script used by the closed-loop workflow, ensuring that the
selected CatBoost features have identical definitions.

Required local files
--------------------
1. Step-2 training script, normally:
       ./02_train_fp_catboost_cycle.py
   Alternative names can be listed in STEP2_SCRIPT_CANDIDATES below.

2. Experimental table:
       ./data/output_with_molar_amount_pubchem.xlsx

3. Compound-SMILES workbook:
       ./data/compound_smiles.xlsx

4. Target-specific CatBoost feature lists:
       ./cd_catboost_feature_selection/Ex_max_nm_best_top20_feature_list.csv
       ./cd_catboost_feature_selection/Em_max_nm_best_top35_feature_list.csv
       ./cd_catboost_feature_selection/QY_%_best_top35_feature_list.csv

5. Existing project module:

Dependencies
------------
    pip install numpy pandas scipy scikit-learn matplotlib openpyxl
    pip install torch catboost rdkit

Run
---
    python ensemble_fpnn_catboost_bo_reviewer_benchmark_complete.py

Start with RUN_MODE = "smoke". After confirming paths and features, switch to
RUN_MODE = "publication" for the final analysis. Publication mode is expensive
because the exact models are repeatedly retrained.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any
import importlib.util
import json
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from catboost import CatBoostRegressor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# USER SETTINGS
# =============================================================================

RUN_MODE = "standard"  # "smoke", "standard", or "publication"
TASK_MODE = "color"  # "color" or "global"
RUN_INITIAL_SIZE_STUDY = True

EXPERIMENT_FILE = "./data/output_with_molar_amount_pubchem.xlsx"
SMILES_PATH = "./data/compound_smiles.xlsx"

STEP2_SCRIPT_CANDIDATES = (
    "./02_train_fp_catboost_cycle.py",
    "./02_train_fp_catboost_cycle_success.py",
    "./validation/02_train_fp_catboost_cycle_success_selected_features.py",
)

FEATURE_LIST_FILES = {
    "Ex_max_nm": "./cd_catboost_feature_selection/Ex_max_nm_best_top20_feature_list.csv",
    "Em_max_nm": "./cd_catboost_feature_selection/Em_max_nm_best_top35_feature_list.csv",
    "QY_%": "./cd_catboost_feature_selection/QY_%_best_top35_feature_list.csv",
}

TARGET_NAMES = ["Ex_max_nm", "Em_max_nm", "QY_%"]
FORBIDDEN_FEATURES = {
    "Ex_max_nm", "Em_max_nm", "QY_%",
    "stoke_shift", "stoke_shift_calc_check", "Stokes_shift_nm",
}

# Actual BO color definitions. These match the later BO script.
COLOR_WINDOWS = {
    "Blue": (440.0, 480.0),
    "Green": (500.0, 540.0),
    "Yellow": (560.0, 580.0),
    "Orange": (580.0, 630.0),
}

# The BO candidate generator uses 15 mL. Existing experimental rows should keep
# their measured reagent_1_mL. This value is used only when that column is absent.
DEFAULT_REAGENT_1_ML = 15.0

# The actual model uses equal numbers of NN and CatBoost members.
NN_FAMILY_WEIGHT = 0.5
CB_FAMILY_WEIGHT = 0.5


@dataclass
class BenchmarkCfg:
    random_seed: int = 42

    # Paired retrospective benchmark.
    n_repeats: int = 3
    initial_size: int = 20
    budget: int = 10

    # Exact ensemble seeds used in the closed-loop training code.
    seeds: Tuple[int, ...] = (42, 2024, 3407, 777, 1024)

    # Exact FPProcessNN training hyperparameters.
    nn_epochs: int = 60
    nn_lr: float = 2e-3
    nn_weight_decay: float = 1e-4
    nn_batch_size: int = 64
    nn_grad_clip: float = 1.0
    nn_predict_batch_size: int = 512

    # Exact CatBoost family/hyperparameter form.
    cb_iterations: int = 300
    cb_learning_rate: float = 0.05
    cb_depth: int = 4
    cb_l2_leaf_reg: float = 5.0
    cb_random_strength: float = 1.0
    cb_subsample: float = 0.9

    # Actual color BO objective.
    alpha: float = 0.7
    color_softness_nm: float = 15.0
    qy_min_percent: float = 0.0
    qy_max_percent: float = 100.0
    stokes_min_nm: float = 40.0
    stokes_max_nm: float = 180.0
    ucb_beta: float = 0.2

    # Optional global score for supplementary analysis.
    global_qy_weight: float = 0.6
    global_stokes_weight: float = 0.4

    top_quantile: float = 0.10
    property_train_success_only: bool = True
    success_positive_threshold: float = 0.5

    # Statistical/plot settings.
    save_dir: str = "./ensemble_fpnn_catboost_reviewer_results"
    dpi: int = 300

    # Main reviewer-oriented comparisons.
    strategies: Tuple[str, ...] = (
        "Random",
        "Chemistry_heuristic",
        "CatBoost_greedy",
        "FPProcessNN_greedy",
        "Ensemble_greedy",
        "Ensemble_UCB",
    )

    # Initial-data-size study.
    initial_sizes: Tuple[int, ...] = (20, 40, 60, 80)
    initial_size_repeats: int = 3
    initial_size_budget: int = 5

    # Runtime behavior.
    device: str = "auto"  # "auto", "cpu", or "cuda"
    deterministic_torch: bool = False
    cache_model_predictions_within_repeat: bool = True


def make_cfg(run_mode: str) -> BenchmarkCfg:
    mode = run_mode.strip().lower()
    if mode == "smoke":
        return BenchmarkCfg(
            n_repeats=3,
            initial_size=20,
            budget=10,
            seeds=(42, 2024),
            nn_epochs=60,
            cb_iterations=300,
            initial_size_repeats=2,
            initial_size_budget=10,
        )
    if mode == "standard":
        return BenchmarkCfg(
            n_repeats=20,
            initial_size=20,
            budget=10,
            seeds=(42, 2024, 3407, 777, 1024),
            nn_epochs=300,
            cb_iterations=2000,
            initial_size_repeats=10,
            initial_size_budget=10,
        )
    if mode == "publication":
        return BenchmarkCfg(
            n_repeats=100,
            initial_size=20,
            budget=20,
            seeds=(42, 2024, 3407, 777, 1024),
            nn_epochs=300,
            cb_iterations=2000,
            initial_size_repeats=30,
            initial_size_budget=20,
        )
    raise ValueError("RUN_MODE must be 'smoke', 'standard', or 'publication'.")


# =============================================================================
# UTILITIES
# =============================================================================


def set_seed(seed: int, deterministic_torch: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(cfg: BenchmarkCfg) -> torch.device:
    requested = cfg.device.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cfg.device='cuda' but CUDA is not available.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("cfg.device must be 'auto', 'cpu', or 'cuda'.")
    return torch.device(requested)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def load_step2_module():
    selected = None
    for candidate in STEP2_SCRIPT_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            selected = p
            break
    if selected is None:
        raise FileNotFoundError(
            "Cannot find the Step-2 model/feature script. Checked:\n  - "
            + "\n  - ".join(STEP2_SCRIPT_CANDIDATES)
            + "\nPlace this benchmark in the same directory as the Step-2 script, "
              "or edit STEP2_SCRIPT_CANDIDATES."
        )

    spec = importlib.util.spec_from_file_location("cd_step2_model_module", str(selected))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import specification for {selected}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for name in ("build_compound_feature_table", "build_dataset_features"):
        if not hasattr(module, name):
            raise AttributeError(f"Step-2 script lacks required function: {name}")

    print(f"Using Step-2 feature module: {selected.resolve()}")
    return module, selected


def read_feature_list(path: str) -> List[str]:
    df = read_table(path)
    if "feature" not in df.columns:
        raise ValueError(f"Feature-list file must contain column 'feature': {path}")
    features = list(dict.fromkeys(df["feature"].dropna().astype(str).str.strip().tolist()))
    leakage = [f for f in features if f in FORBIDDEN_FEATURES]
    if leakage:
        raise ValueError(f"Feature list contains target leakage columns: {leakage}")
    if not features:
        raise ValueError(f"No features found in {path}")
    return features


def ensure_shift_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "stoke_shift" not in out.columns:
        if "Stokes_shift_nm" in out.columns:
            out["stoke_shift"] = out["Stokes_shift_nm"]
        elif "Ex_max_nm" in out.columns and "Em_max_nm" in out.columns:
            out["stoke_shift"] = out["Em_max_nm"] - out["Ex_max_nm"]
    if "Stokes_shift_nm" not in out.columns:
        if "stoke_shift" in out.columns:
            out["Stokes_shift_nm"] = out["stoke_shift"]
        elif "Ex_max_nm" in out.columns and "Em_max_nm" in out.columns:
            out["Stokes_shift_nm"] = out["Em_max_nm"] - out["Ex_max_nm"]
    return out


def ensure_recipe_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Make the historical table compatible with Step-2 feature engineering."""
    out = df.copy()

    if "reagent" in out.columns and "reagent_1" not in out.columns:
        out = out.rename(columns={"reagent": "reagent_1"})

    if "mass_1" in out.columns and "mass_1_g" not in out.columns:
        out = out.rename(columns={"mass_1": "mass_1_g"})
    if "mass_2" in out.columns and "mass_2_g" not in out.columns:
        out = out.rename(columns={"mass_2": "mass_2_g"})

    if "Mass1(g)" not in out.columns:
        if "mass_1_g" in out.columns:
            out["Mass1(g)"] = out["mass_1_g"]
        elif "mass_1_mg" in out.columns:
            out["Mass1(g)"] = pd.to_numeric(out["mass_1_mg"], errors="coerce") / 1000.0
    if "Mass2(g)" not in out.columns:
        if "mass_2_g" in out.columns:
            out["Mass2(g)"] = out["mass_2_g"]
        elif "mass_2_mg" in out.columns:
            out["Mass2(g)"] = pd.to_numeric(out["mass_2_mg"], errors="coerce") / 1000.0

    if "mass_1_g" not in out.columns and "Mass1(g)" in out.columns:
        out["mass_1_g"] = out["Mass1(g)"]
    if "mass_2_g" not in out.columns and "Mass2(g)" in out.columns:
        out["mass_2_g"] = out["Mass2(g)"]

    if "reagent_1_mL" not in out.columns:
        out["reagent_1_mL"] = DEFAULT_REAGENT_1_ML
        warnings.warn(
            f"reagent_1_mL was absent; filled with {DEFAULT_REAGENT_1_ML} mL. "
            "Replace this with measured values if the historical volume varied."
        )

    required = [
                   "precursor_1", "precursor_2", "reagent_1",
                   "pre_1_mmol", "pre_2_mmol", "Mass1(g)", "Mass2(g)",
                   "reagent_1_mL", "temperature", "time_h",
               ] + TARGET_NAMES
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            "Historical table is missing columns required by the exact model: "
            f"{missing}"
        )
    return out


def get_solvent_polarity_data() -> Dict[str, Dict[str, float]]:
    return {
        "water": {"dielectric_constant": 80.1, "pi_star": 1.09, "HBD": 1.17, "HBA": 0.47, "ET30": 63.1, "pKa": 15.7,
                  "ORP_mV": 300, "density_g_cm3": 0.997, "specific_heat_J_gK": 4.18, "viscosity_mPa_s": 0.89,
                  "boiling_point_C": 100},
        "dimethylformamide": {"dielectric_constant": 36.7, "pi_star": 0.88, "HBD": 0.00, "HBA": 0.69, "ET30": 43.2,
                              "pKa": 29, "ORP_mV": 200, "density_g_cm3": 0.944, "specific_heat_J_gK": 2.47,
                              "viscosity_mPa_s": 0.92, "boiling_point_C": 153},
        "ethanol": {"dielectric_constant": 24.6, "pi_star": 0.54, "HBD": 0.86, "HBA": 0.75, "ET30": 51.9, "pKa": 15.9,
                    "ORP_mV": 250, "density_g_cm3": 0.789, "specific_heat_J_gK": 2.44, "viscosity_mPa_s": 1.07,
                    "boiling_point_C": 78.37},
        "sulfuric acid": {"dielectric_constant": 83, "pi_star": 1.11, "HBD": 1.22, "HBA": 0.48, "ET30": 64.8,
                          "pKa": 0.0, "ORP_mV": 800, "density_g_cm3": 1.05, "specific_heat_J_gK": 3.6,
                          "viscosity_mPa_s": 1.2, "boiling_point_C": 100},
        "phosphoric acid": {"dielectric_constant": 82, "pi_star": 1.10, "HBD": 1.20, "HBA": 0.47, "ET30": 64.0,
                            "pKa": 1.0, "ORP_mV": 700, "density_g_cm3": 1.03, "specific_heat_J_gK": 3.8,
                            "viscosity_mPa_s": 1.1, "boiling_point_C": 100},
    }


# =============================================================================
# EXACT FPProcessNN ARCHITECTURE
# =============================================================================


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
        self.fp_encoder = FingerprintEncoderSmall()
        self.proc_encoder = ProcessEncoderSmall()
        self.fusion_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(64, 64),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(80, 64),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(64, 3),
        )

    def forward(
            self,
            fps_1: torch.Tensor,
            fps_2: torch.Tensor,
            global_feat: torch.Tensor,
    ) -> torch.Tensor:
        f1 = self.fp_encoder(fps_1)
        f2 = self.fp_encoder(fps_2)
        fusion = self.fusion_mlp(torch.cat([f1 + f2, torch.abs(f1 - f2)], dim=-1))
        proc = self.proc_encoder(global_feat)
        return self.head(torch.cat([fusion, proc], dim=-1))


class TensorRecipeDataset(Dataset):
    def __init__(
            self,
            fps_1: torch.Tensor,
            fps_2: torch.Tensor,
            global_feat: torch.Tensor,
            y: torch.Tensor,
            indices: Sequence[int],
    ):
        self.fps_1 = fps_1
        self.fps_2 = fps_2
        self.global_feat = global_feat
        self.y = y
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i = int(self.indices[idx])
        return {
            "fps_1": self.fps_1[i],
            "fps_2": self.fps_2[i],
            "global_feat": self.global_feat[i],
            "y": self.y[i],
        }


def collate_fn(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "fps_1": torch.stack([b["fps_1"] for b in batch]),
        "fps_2": torch.stack([b["fps_2"] for b in batch]),
        "global_feat": torch.stack([b["global_feat"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


# =============================================================================
# PRECOMPUTED DATA BUNDLE
# =============================================================================


@dataclass
class ModelDataBundle:
    df: pd.DataFrame
    y: np.ndarray
    fps_1: torch.Tensor
    fps_2: torch.Tensor
    global_feat: torch.Tensor
    cb_X: Dict[str, np.ndarray]
    cb_feature_names: Dict[str, List[str]]
    space_X: np.ndarray
    chemistry_heuristic: np.ndarray


def load_and_prepare_data(cfg: BenchmarkCfg) -> ModelDataBundle:
    from model.model import data_process, CarbonDotFeaturizer
    step2, step2_path = load_step2_module()

    raw = read_table(EXPERIMENT_FILE)
    raw = ensure_shift_columns(raw)
    processed = data_process(raw).reset_index(drop=True)
    processed = ensure_shift_columns(processed)
    processed = ensure_recipe_aliases(processed)

    # Keep only rows with complete measured targets and recipe fields.
    required = [
                   "precursor_1", "precursor_2", "reagent_1",
                   "temperature", "time_h", "pre_1_mmol", "pre_2_mmol",
                   "Mass1(g)", "Mass2(g)", "reagent_1_mL",
               ] + TARGET_NAMES
    processed = processed.dropna(subset=required).reset_index(drop=True)

    if cfg.property_train_success_only and "success" in processed.columns:
        success = pd.to_numeric(processed["success"], errors="coerce")
        mask = success >= cfg.success_positive_threshold
        if int(mask.sum()) >= cfg.initial_size + cfg.budget:
            processed = processed.loc[mask].reset_index(drop=True)
        else:
            warnings.warn(
                "Too few success=1 rows for the requested benchmark; all measured rows are used."
            )

    if len(processed) < cfg.initial_size + cfg.budget:
        raise ValueError(
            f"Usable rows={len(processed)}, but initial_size+budget="
            f"{cfg.initial_size + cfg.budget}."
        )

    # Exact NN input pipeline.
    smiles_df = pd.read_excel(SMILES_PATH)
    featurizer = CarbonDotFeaturizer(smiles_df, get_solvent_polarity_data())

    precursor_names = sorted(
        set(processed["precursor_1"].astype(str))
        | set(processed["precursor_2"].astype(str))
    )
    fp_map = {
        p: torch.tensor(featurizer.fingerprints(p), dtype=torch.float32)
        for p in precursor_names
    }
    reagent_names = sorted(set(processed["reagent_1"].astype(str)))
    solvent_map = {r: featurizer.solvent_feat(r).float() for r in reagent_names}

    fps1_rows: List[torch.Tensor] = []
    fps2_rows: List[torch.Tensor] = []
    global_rows: List[torch.Tensor] = []

    for _, row in processed.iterrows():
        p1 = str(row["precursor_1"])
        p2 = str(row["precursor_2"])
        reagent = str(row["reagent_1"])
        pre1 = float(row["pre_1_mmol"])
        pre2 = float(row["pre_2_mmol"])
        ratio = pre1 / (pre1 + pre2) if (pre1 + pre2) > 0 else 0.5

        s_feat = solvent_map[reagent]
        r_feat = featurizer.reaction_feat(row["temperature"], row["time_h"]).float()
        g = torch.cat([s_feat, r_feat, torch.tensor([ratio], dtype=torch.float32)])

        fps1_rows.append(fp_map[p1])
        fps2_rows.append(fp_map[p2])
        global_rows.append(g)

    fps_1 = torch.stack(fps1_rows)
    fps_2 = torch.stack(fps2_rows)
    global_feat = torch.stack(global_rows)

    if fps_1.shape[1] != 679:
        raise ValueError(
            f"FPProcessNN expects fingerprint dimension 679, got {fps_1.shape[1]}. "
            "Check CarbonDotFeaturizer and the compound SMILES table."
        )
    if global_feat.shape[1] != 14:
        raise ValueError(
            f"FPProcessNN expects process/global dimension 14, got {global_feat.shape[1]}. "
            "Check solvent_feat(), reaction_feat(), and precursor ratio construction."
        )

    y = processed[TARGET_NAMES].astype(float).values.astype(np.float32)

    # Exact CatBoost feature engineering from the supplied Step-2 script.
    try:
        smiles_compound = pd.read_excel(SMILES_PATH, sheet_name="compound")
    except Exception:
        smiles_compound = pd.read_excel(SMILES_PATH)

    compound_feat_df, fp_objs = step2.build_compound_feature_table(smiles_compound)
    feature_df = step2.build_dataset_features(
        processed.copy(), compound_feat_df, fp_objs, has_targets=True
    )

    if len(feature_df) != len(processed):
        raise ValueError(
            f"CatBoost feature rows ({len(feature_df)}) do not match processed rows "
            f"({len(processed)})."
        )

    cb_X: Dict[str, np.ndarray] = {}
    cb_feature_names: Dict[str, List[str]] = {}

    for target in TARGET_NAMES:
        features = read_feature_list(FEATURE_LIST_FILES[target])
        missing = [c for c in features if c not in feature_df.columns]
        if missing:
            raise ValueError(
                f"CatBoost feature table lacks selected features for {target}: {missing[:30]}"
            )
        X_df = feature_df[features].copy()
        for col in features:
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce").fillna(0.0)
        cb_X[target] = X_df.values.astype(np.float32)
        cb_feature_names[target] = features

    # A leakage-free representation for space filling: union of selected CB inputs.
    union_features = list(dict.fromkeys(
        cb_feature_names["Ex_max_nm"]
        + cb_feature_names["Em_max_nm"]
        + cb_feature_names["QY_%"]
    ))
    space_df = feature_df[union_features].copy()
    for col in union_features:
        space_df[col] = pd.to_numeric(space_df[col], errors="coerce").fillna(0.0)
    space_X = space_df.values.astype(np.float32)

    chemistry_heuristic = compute_chemistry_heuristic(feature_df, processed)

    print(f"Prepared {len(processed)} historical recipes.")
    print(f"NN fingerprint dimension: {fps_1.shape[1]}")
    print(f"NN process/global dimension: {global_feat.shape[1]}")
    print("CatBoost selected feature counts: " + ", ".join(
        f"{t}={len(cb_feature_names[t])}" for t in TARGET_NAMES
    ))
    print(f"Step-2 source: {step2_path}")

    return ModelDataBundle(
        df=processed,
        y=y,
        fps_1=fps_1,
        fps_2=fps_2,
        global_feat=global_feat,
        cb_X=cb_X,
        cb_feature_names=cb_feature_names,
        space_X=space_X,
        chemistry_heuristic=chemistry_heuristic,
    )


def compute_chemistry_heuristic(feature_df: pd.DataFrame, recipe_df: pd.DataFrame) -> np.ndarray:
    """
    Transparent chemistry-inspired baseline. It does not use Ex, Em, QY, Stokes,
    predictions, or any target-derived feature. It is not labeled as a human
    expert; a genuine chemist baseline should be prospectively preregistered.
    """
    n = len(feature_df)

    def numeric(name: str, default: float = 0.0) -> np.ndarray:
        if name not in feature_df.columns:
            return np.full(n, default, dtype=float)
        return pd.to_numeric(feature_df[name], errors="coerce").fillna(default).values.astype(float)

    def minmax(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        span = np.nanmax(v) - np.nanmin(v)
        if not np.isfinite(span) or span < 1e-12:
            return np.zeros_like(v)
        return (v - np.nanmin(v)) / span

    hetero = (
            numeric("pair_total_N")
            + numeric("pair_total_O")
            + numeric("pair_total_S")
            + numeric("pair_total_P")
    )
    aromatic = numeric("pair_sum_NumAromaticRings")
    amine_acid = numeric("pair_amine_plus_acid")
    tanimoto_diversity = 1.0 - np.clip(numeric("fp_tanimoto"), 0.0, 1.0)

    temp = pd.to_numeric(recipe_df["temperature"], errors="coerce").fillna(0).values.astype(float)
    time_h = pd.to_numeric(recipe_df["time_h"], errors="coerce").fillna(0).values.astype(float)
    pre1 = pd.to_numeric(recipe_df["pre_1_mmol"], errors="coerce").fillna(0).values.astype(float)
    pre2 = pd.to_numeric(recipe_df["pre_2_mmol"], errors="coerce").fillna(0).values.astype(float)
    frac = np.divide(pre1, pre1 + pre2, out=np.full_like(pre1, 0.5), where=(pre1 + pre2) > 0)
    balanced_ratio = 1.0 - np.clip(np.abs(frac - 0.5) / 0.5, 0.0, 1.0)

    temp_n = minmax(temp)
    time_n = minmax(time_h)
    temp_mid = 1.0 - np.abs(temp_n - 0.5) * 2.0
    time_mid = 1.0 - np.abs(time_n - 0.5) * 2.0

    score = (
            1.5 * minmax(hetero)
            + 1.0 * minmax(aromatic)
            + 0.8 * minmax(amine_acid)
            + 0.5 * tanimoto_diversity
            + 0.8 * balanced_ratio
            + 0.5 * temp_mid
            + 0.3 * time_mid
    )
    return score.astype(float)


# =============================================================================
# EXACT MODEL RETRAINING AND PREDICTION
# =============================================================================


def train_fpnn_ensemble(
        bundle: ModelDataBundle,
        labeled_idx: Sequence[int],
        cfg: BenchmarkCfg,
        device: torch.device,
) -> Tuple[List[FPProcessNN], StandardScaler]:
    labeled_idx = np.asarray(labeled_idx, dtype=int)
    y_train = bundle.y[labeled_idx].astype(np.float32)
    y_scaler = StandardScaler().fit(y_train)

    dataset = TensorRecipeDataset(
        bundle.fps_1,
        bundle.fps_2,
        bundle.global_feat,
        torch.tensor(bundle.y, dtype=torch.float32),
        labeled_idx,
    )

    models: List[FPProcessNN] = []
    for seed in cfg.seeds:
        set_seed(int(seed), cfg.deterministic_torch)
        model = FPProcessNN().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.nn_lr,
            weight_decay=cfg.nn_weight_decay,
        )
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loader = DataLoader(
            dataset,
            batch_size=min(cfg.nn_batch_size, max(1, len(dataset))),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
            generator=generator,
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
                target = torch.tensor(y_scaled, dtype=torch.float32, device=device)
                loss = F.mse_loss(pred, target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.nn_grad_clip)
                optimizer.step()

        model.eval()
        models.append(model)

    return models, y_scaler


def predict_fpnn_members(
        models: Sequence[FPProcessNN],
        y_scaler: StandardScaler,
        bundle: ModelDataBundle,
        pool_idx: Sequence[int],
        cfg: BenchmarkCfg,
        device: torch.device,
) -> np.ndarray:
    pool_idx = np.asarray(pool_idx, dtype=int)
    fp1 = bundle.fps_1[pool_idx]
    fp2 = bundle.fps_2[pool_idx]
    glob = bundle.global_feat[pool_idx]

    member_predictions: List[np.ndarray] = []
    for model in models:
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(pool_idx), cfg.nn_predict_batch_size):
                end = start + cfg.nn_predict_batch_size
                pred_scaled = model(
                    fp1[start:end].to(device),
                    fp2[start:end].to(device),
                    glob[start:end].to(device),
                ).detach().cpu().numpy()
                chunks.append(pred_scaled)
        pred_scaled_all = np.concatenate(chunks, axis=0)
        pred = y_scaler.inverse_transform(pred_scaled_all)
        member_predictions.append(pred.astype(np.float32))

    return np.stack(member_predictions, axis=0)  # [seed, pool, 3]


def make_catboost_model(cfg: BenchmarkCfg, random_seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=cfg.cb_iterations,
        learning_rate=cfg.cb_learning_rate,
        depth=cfg.cb_depth,
        l2_leaf_reg=cfg.cb_l2_leaf_reg,
        random_strength=cfg.cb_random_strength,
        bootstrap_type="Bernoulli",
        subsample=cfg.cb_subsample,
        grow_policy="SymmetricTree",
        random_seed=int(random_seed),
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def train_predict_catboost_members(
        bundle: ModelDataBundle,
        labeled_idx: Sequence[int],
        pool_idx: Sequence[int],
        cfg: BenchmarkCfg,
) -> np.ndarray:
    labeled_idx = np.asarray(labeled_idx, dtype=int)
    pool_idx = np.asarray(pool_idx, dtype=int)

    target_member_predictions: List[np.ndarray] = []
    for target_index, target in enumerate(TARGET_NAMES):
        X_all = bundle.cb_X[target]
        X_train = X_all[labeled_idx]
        X_pool = X_all[pool_idx]
        y_train = bundle.y[labeled_idx, target_index]

        predictions_this_target: List[np.ndarray] = []
        for seed_index, seed in enumerate(cfg.seeds):
            # Exact Step-2 convention: seed + 1000 * model index.
            model_seed = int(seed) + 1000 * seed_index
            model = make_catboost_model(cfg, model_seed)
            model.fit(X_train, y_train, verbose=False)
            pred = np.asarray(model.predict(X_pool), dtype=np.float32)
            predictions_this_target.append(pred)

        target_member_predictions.append(
            np.stack(predictions_this_target, axis=0)
        )

    # [target, seed, pool] -> [seed, pool, target]
    return np.stack(target_member_predictions, axis=-1).astype(np.float32)


@dataclass
class PredictionSet:
    nn_members: np.ndarray  # [seed, pool, 3]
    cb_members: np.ndarray  # [seed, pool, 3]
    ensemble_members: np.ndarray  # [2*seed, pool, 3]
    nn_mean: np.ndarray
    nn_std: np.ndarray
    cb_mean: np.ndarray
    cb_std: np.ndarray
    ensemble_mean: np.ndarray
    ensemble_std: np.ndarray


def train_predict_exact_ensemble(
        bundle: ModelDataBundle,
        labeled_idx: Sequence[int],
        pool_idx: Sequence[int],
        cfg: BenchmarkCfg,
        device: torch.device,
) -> PredictionSet:
    nn_models, y_scaler = train_fpnn_ensemble(bundle, labeled_idx, cfg, device)
    nn_members = predict_fpnn_members(
        nn_models, y_scaler, bundle, pool_idx, cfg, device
    )
    cb_members = train_predict_catboost_members(
        bundle, labeled_idx, pool_idx, cfg
    )

    if nn_members.shape != cb_members.shape:
        raise ValueError(
            f"NN and CatBoost member prediction shapes differ: "
            f"{nn_members.shape} vs {cb_members.shape}. "
            "The exact 0.5/0.5 implementation requires equal member counts."
        )

    ensemble_members = np.concatenate([nn_members, cb_members], axis=0)
    nn_mean = nn_members.mean(axis=0)
    nn_std = nn_members.std(axis=0)
    cb_mean = cb_members.mean(axis=0)
    cb_std = cb_members.std(axis=0)
    ensemble_mean = ensemble_members.mean(axis=0)
    ensemble_std = ensemble_members.std(axis=0)

    # Numerical verification of Ensemble_mean_0p5NN_0p5CB.
    expected = NN_FAMILY_WEIGHT * nn_mean + CB_FAMILY_WEIGHT * cb_mean
    if not np.allclose(ensemble_mean, expected, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Concatenated-member mean does not equal 0.5NN + 0.5CB.")

    # Free GPU model memory after predictions are created.
    del nn_models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return PredictionSet(
        nn_members=nn_members,
        cb_members=cb_members,
        ensemble_members=ensemble_members,
        nn_mean=nn_mean,
        nn_std=nn_std,
        cb_mean=cb_mean,
        cb_std=cb_std,
        ensemble_mean=ensemble_mean,
        ensemble_std=ensemble_std,
    )


# =============================================================================
# OBJECTIVE AND ACQUISITION
# =============================================================================


def fixed_minmax(x: np.ndarray, lower: float, upper: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if upper <= lower:
        raise ValueError(f"Invalid bounds: lower={lower}, upper={upper}")
    return np.clip((x - lower) / (upper - lower), 0.0, 1.0)


def color_fit_array(
        em: np.ndarray,
        lower: float,
        upper: float,
        softness_nm: float,
) -> np.ndarray:
    em = np.asarray(em, dtype=float)
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    outside = np.maximum(np.abs(em - center) - half_width, 0.0)
    return np.exp(-(outside ** 2) / (2.0 * softness_nm ** 2))


def observed_true_score(
        y: np.ndarray,
        task_name: str,
        cfg: BenchmarkCfg,
) -> np.ndarray:
    ex = y[:, 0].astype(float)
    em = y[:, 1].astype(float)
    qy = y[:, 2].astype(float)
    stokes = em - ex

    qy_score = fixed_minmax(qy, cfg.qy_min_percent, cfg.qy_max_percent)
    stokes_score = fixed_minmax(stokes, cfg.stokes_min_nm, cfg.stokes_max_nm)

    if task_name == "GLOBAL":
        return (
                cfg.global_qy_weight * qy_score
                + cfg.global_stokes_weight * stokes_score
        )

    lower, upper = COLOR_WINDOWS[task_name]
    fit = color_fit_array(em, lower, upper, cfg.color_softness_nm)
    return fit * (
            cfg.alpha * qy_score
            + (1.0 - cfg.alpha) * stokes_score
    )


def predicted_score_mu_sigma(
        pred_mean: np.ndarray,
        pred_std: np.ndarray,
        task_name: str,
        cfg: BenchmarkCfg,
) -> Tuple[np.ndarray, np.ndarray]:
    ex = pred_mean[:, 0].astype(float)
    em = pred_mean[:, 1].astype(float)
    qy = pred_mean[:, 2].astype(float)
    ex_std = pred_std[:, 0].astype(float)
    em_std = pred_std[:, 1].astype(float)
    qy_std = pred_std[:, 2].astype(float)

    stokes = em - ex
    stokes_std = np.sqrt(ex_std ** 2 + em_std ** 2)

    qy_score = fixed_minmax(qy, cfg.qy_min_percent, cfg.qy_max_percent)
    stokes_score = fixed_minmax(stokes, cfg.stokes_min_nm, cfg.stokes_max_nm)

    qy_std_norm = qy_std / max(
        cfg.qy_max_percent - cfg.qy_min_percent, 1e-8
    )
    stokes_std_norm = stokes_std / max(
        cfg.stokes_max_nm - cfg.stokes_min_nm, 1e-8
    )

    if task_name == "GLOBAL":
        score_mu = (
                cfg.global_qy_weight * qy_score
                + cfg.global_stokes_weight * stokes_score
        )
        score_sigma = (
                cfg.global_qy_weight * qy_std_norm
                + cfg.global_stokes_weight * stokes_std_norm
        )
        return score_mu, score_sigma

    lower, upper = COLOR_WINDOWS[task_name]
    fit = color_fit_array(em, lower, upper, cfg.color_softness_nm)
    score_mu = fit * (
            cfg.alpha * qy_score
            + (1.0 - cfg.alpha) * stokes_score
    )
    score_sigma = fit * (
            cfg.alpha * qy_std_norm
            + (1.0 - cfg.alpha) * stokes_std_norm
    )
    return score_mu, score_sigma


def task_names(task_mode: str) -> List[str]:
    mode = task_mode.strip().lower()
    if mode == "color":
        return list(COLOR_WINDOWS.keys())
    if mode == "global":
        return ["GLOBAL"]
    raise ValueError("TASK_MODE must be 'color' or 'global'.")


# =============================================================================
# RETROSPECTIVE SEARCH
# =============================================================================


def pareto_mask(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    mask = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        if not mask[i]:
            continue
        dominated = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        if np.any(dominated):
            mask[i] = False
    return mask


def hypervolume_2d(
        points: np.ndarray,
        ref: Tuple[float, float] = (0.0, 0.0),
) -> float:
    if len(points) == 0:
        return 0.0
    pts = np.asarray(points, dtype=float)
    pts = pts[(pts[:, 0] > ref[0]) & (pts[:, 1] > ref[1])]
    if len(pts) == 0:
        return 0.0
    pts = pts[pareto_mask(pts)]
    pts = pts[np.argsort(-pts[:, 0])]

    hv = 0.0
    prev_x = 1.0
    current_y = ref[1]
    for x, y in pts:
        hv += max(prev_x - x, 0.0) * max(current_y - ref[1], 0.0)
        current_y = max(current_y, y)
        prev_x = x
    hv += max(prev_x - ref[0], 0.0) * max(current_y - ref[1], 0.0)
    return float(hv)


def select_space_filling(
        bundle: ModelDataBundle,
        labeled: Sequence[int],
        pool: Sequence[int],
) -> int:
    labeled = np.asarray(labeled, dtype=int)
    pool = np.asarray(pool, dtype=int)
    scaler = StandardScaler().fit(bundle.space_X[labeled])
    X_labeled = scaler.transform(bundle.space_X[labeled])
    X_pool = scaler.transform(bundle.space_X[pool])
    distances = pairwise_distances(X_pool, X_labeled, metric="euclidean")
    return int(pool[np.argmax(distances.min(axis=1))])


def choose_from_predictions(
        strategy: str,
        predictions: PredictionSet,
        pool: Sequence[int],
        task_name: str,
        cfg: BenchmarkCfg,
) -> int:
    pool = np.asarray(pool, dtype=int)

    if strategy == "CatBoost_greedy":
        mean, std = predictions.cb_mean, predictions.cb_std
        score_mu, _ = predicted_score_mu_sigma(mean, std, task_name, cfg)
        acquisition = score_mu
    elif strategy == "FPProcessNN_greedy":
        mean, std = predictions.nn_mean, predictions.nn_std
        score_mu, _ = predicted_score_mu_sigma(mean, std, task_name, cfg)
        acquisition = score_mu
    elif strategy == "Ensemble_greedy":
        mean, std = predictions.ensemble_mean, predictions.ensemble_std
        score_mu, _ = predicted_score_mu_sigma(mean, std, task_name, cfg)
        acquisition = score_mu
    elif strategy == "Ensemble_UCB":
        mean, std = predictions.ensemble_mean, predictions.ensemble_std
        score_mu, score_sigma = predicted_score_mu_sigma(mean, std, task_name, cfg)
        acquisition = score_mu + cfg.ucb_beta * score_sigma
    else:
        raise ValueError(f"Unsupported model strategy: {strategy}")

    acquisition = np.asarray(acquisition, dtype=float)
    acquisition[~np.isfinite(acquisition)] = -np.inf
    if np.all(~np.isfinite(acquisition)):
        raise RuntimeError(f"All acquisition values are non-finite for {strategy}.")
    return int(pool[np.argmax(acquisition)])


class RepeatPredictionCache:
    """Cache only within one repeat; prevents unbounded memory growth."""

    def __init__(self):
        self._cache: Dict[Tuple[int, ...], Tuple[np.ndarray, PredictionSet]] = {}

    def get(self, labeled: Sequence[int]) -> Optional[Tuple[np.ndarray, PredictionSet]]:
        return self._cache.get(tuple(sorted(map(int, labeled))))

    def set(
            self,
            labeled: Sequence[int],
            pool: Sequence[int],
            predictions: PredictionSet,
    ) -> None:
        self._cache[tuple(sorted(map(int, labeled)))] = (
            np.asarray(pool, dtype=int), predictions
        )


def get_exact_predictions(
        bundle: ModelDataBundle,
        labeled: Sequence[int],
        pool: Sequence[int],
        cfg: BenchmarkCfg,
        device: torch.device,
        cache: RepeatPredictionCache,
) -> PredictionSet:
    if cfg.cache_model_predictions_within_repeat:
        cached = cache.get(labeled)
        if cached is not None:
            cached_pool, pred = cached
            if np.array_equal(cached_pool, np.asarray(pool, dtype=int)):
                return pred

    predictions = train_predict_exact_ensemble(
        bundle=bundle,
        labeled_idx=labeled,
        pool_idx=pool,
        cfg=cfg,
        device=device,
    )
    if cfg.cache_model_predictions_within_repeat:
        cache.set(labeled, pool, predictions)
    return predictions


def run_one_strategy(
        strategy: str,
        bundle: ModelDataBundle,
        true_score: np.ndarray,
        task_name: str,
        initial_indices: Sequence[int],
        candidate_indices: Sequence[int],
        cfg: BenchmarkCfg,
        device: torch.device,
        rng_seed: int,
        cache: RepeatPredictionCache,
) -> Tuple[pd.DataFrame, List[int]]:
    rng = np.random.default_rng(rng_seed)
    labeled = list(map(int, initial_indices))
    pool = list(map(int, candidate_indices))

    threshold = float(np.quantile(true_score, 1.0 - cfg.top_quantile))
    initial_hits = int(np.sum(true_score[np.asarray(labeled)] >= threshold))

    qy_norm_true = fixed_minmax(
        bundle.y[:, 2], cfg.qy_min_percent, cfg.qy_max_percent
    )
    stokes_true = bundle.y[:, 1] - bundle.y[:, 0]
    stokes_norm_true = fixed_minmax(
        stokes_true, cfg.stokes_min_nm, cfg.stokes_max_nm
    )

    rows: List[Dict[str, Any]] = []
    selected_after_initial: List[int] = []

    for step in range(cfg.budget + 1):
        labeled_array = np.asarray(labeled, dtype=int)
        scores = true_score[labeled_array]
        current_hits = int(np.sum(scores >= threshold))
        selected_only_hits = current_hits - initial_hits

        selected_scores = (
            true_score[np.asarray(selected_after_initial, dtype=int)]
            if selected_after_initial else np.asarray([], dtype=float)
        )
        points = np.column_stack([
            qy_norm_true[labeled_array],
            stokes_norm_true[labeled_array],
        ])

        rows.append({
            "task": task_name,
            "strategy": strategy,
            "step": step,
            "n_labeled": len(labeled),
            "best_score": float(np.max(scores)),
            "mean_score_all_labeled": float(np.mean(scores)),
            "mean_score_newly_selected": (
                float(np.mean(selected_scores)) if len(selected_scores) else np.nan
            ),
            "hit_rate_top10_all": float(current_hits / len(labeled)),
            "n_hits_top10_all": current_hits,
            "selected_only_hits_top10": selected_only_hits,
            "selected_only_hit_rate_top10": (
                float(selected_only_hits / step) if step > 0 else np.nan
            ),
            "hypervolume": hypervolume_2d(points),
        })

        if step >= cfg.budget or not pool:
            break

        pool_array = np.asarray(pool, dtype=int)
        if strategy == "Random":
            chosen = int(rng.choice(pool_array))
        elif strategy == "Space_filling":
            chosen = select_space_filling(bundle, labeled, pool)
        elif strategy == "Chemistry_heuristic":
            chosen = int(pool_array[np.argmax(bundle.chemistry_heuristic[pool_array])])
        else:
            predictions = get_exact_predictions(
                bundle, labeled, pool, cfg, device, cache
            )
            chosen = choose_from_predictions(
                strategy, predictions, pool, task_name, cfg
            )

        labeled.append(chosen)
        selected_after_initial.append(chosen)
        pool.remove(chosen)

    return pd.DataFrame(rows), labeled


def run_benchmark(
        bundle: ModelDataBundle,
        cfg: BenchmarkCfg,
        task_mode: str,
        device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_indices = np.arange(len(bundle.y), dtype=int)
    rng = np.random.default_rng(cfg.random_seed)

    if cfg.initial_size + cfg.budget > len(all_indices):
        raise ValueError("initial_size + budget exceeds the available dataset.")

    trajectories: List[pd.DataFrame] = []
    finals: List[Dict[str, Any]] = []

    for task_name in task_names(task_mode):
        true_score = observed_true_score(bundle.y, task_name, cfg)

        for repeat in range(1, cfg.n_repeats + 1):
            initial = rng.choice(
                all_indices,
                size=cfg.initial_size,
                replace=False,
            )
            candidates = np.setdiff1d(all_indices, initial)
            repeat_cache = RepeatPredictionCache()

            for strategy_index, strategy in enumerate(cfg.strategies):
                started = time.time()
                traj, _ = run_one_strategy(
                    strategy=strategy,
                    bundle=bundle,
                    true_score=true_score,
                    task_name=task_name,
                    initial_indices=initial,
                    candidate_indices=candidates,
                    cfg=cfg,
                    device=device,
                    rng_seed=(
                            cfg.random_seed
                            + repeat * 100003
                            + strategy_index * 1009
                    ),
                    cache=repeat_cache,
                )
                traj["repeat"] = repeat
                trajectories.append(traj)

                final_row = traj.iloc[-1].to_dict()
                final_row["repeat"] = repeat
                final_row["runtime_seconds"] = float(time.time() - started)
                finals.append(final_row)

                print(
                    f"Finished task={task_name}, repeat={repeat}/{cfg.n_repeats}, "
                    f"strategy={strategy}, runtime={time.time() - started:.1f}s"
                )

    return pd.concat(trajectories, ignore_index=True), pd.DataFrame(finals)


# =============================================================================
# STATISTICS
# =============================================================================


def make_strategy_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "best_score",
        "mean_score_newly_selected",
        "selected_only_hit_rate_top10",
        "selected_only_hits_top10",
        "hypervolume",
        "runtime_seconds",
    ]
    rows: List[Dict[str, Any]] = []
    for (task, strategy), group in final_df.groupby(["task", "strategy"]):
        row: Dict[str, Any] = {
            "task": task,
            "strategy": strategy,
            "n_repeats": len(group),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_median"] = float(values.median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["task", "best_score_mean"], ascending=[True, False]
    )


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min((len(p) - rank) * p[idx], 1.0)
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def paired_tests(final_df: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("Ensemble_UCB", "Random"),
        ("Ensemble_greedy", "Random"),
        ("Ensemble_UCB", "Space_filling"),
        ("Ensemble_UCB", "Chemistry_heuristic"),
        ("Ensemble_greedy", "CatBoost_greedy"),
        ("Ensemble_greedy", "FPProcessNN_greedy"),
        ("Ensemble_UCB", "Ensemble_greedy"),
    ]
    metrics = [
        "best_score",
        "mean_score_newly_selected",
        "selected_only_hit_rate_top10",
        "selected_only_hits_top10",
        "hypervolume",
    ]

    rows: List[Dict[str, Any]] = []
    for task, task_df in final_df.groupby("task"):
        for strategy_a, strategy_b in comparisons:
            a_df = task_df[task_df["strategy"] == strategy_a].set_index("repeat")
            b_df = task_df[task_df["strategy"] == strategy_b].set_index("repeat")
            common = a_df.index.intersection(b_df.index)
            if len(common) == 0:
                continue

            for metric in metrics:
                a = pd.to_numeric(a_df.loc[common, metric], errors="coerce").values
                b = pd.to_numeric(b_df.loc[common, metric], errors="coerce").values
                valid = np.isfinite(a) & np.isfinite(b)
                a, b = a[valid], b[valid]
                if len(a) == 0:
                    continue
                diff = a - b
                if np.allclose(diff, 0.0):
                    p_value = 1.0
                else:
                    try:
                        p_value = float(
                            wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue
                        )
                    except ValueError:
                        p_value = 1.0

                rows.append({
                    "task": task,
                    "strategy_A": strategy_a,
                    "strategy_B": strategy_b,
                    "metric": metric,
                    "n_pairs": len(a),
                    "mean_A_minus_B": float(np.mean(diff)),
                    "median_A_minus_B": float(np.median(diff)),
                    "fraction_A_better": float(np.mean(diff > 0)),
                    "wilcoxon_p": p_value,
                })

    out = pd.DataFrame(rows)
    if len(out):
        out["holm_p_within_all_tests"] = holm_adjust(out["wilcoxon_p"].values)
    return out


# =============================================================================
# OPTIONAL INITIAL-SIZE STUDY
# =============================================================================


def run_initial_size_study(
        bundle: ModelDataBundle,
        cfg: BenchmarkCfg,
        task_mode: str,
        device: torch.device,
) -> pd.DataFrame:
    all_indices = np.arange(len(bundle.y), dtype=int)
    valid_sizes = [
        size for size in cfg.initial_sizes
        if size >= 10 and size + cfg.initial_size_budget <= len(all_indices)
    ]
    if not valid_sizes:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(cfg.random_seed + 1234567)

    small_cfg = BenchmarkCfg(**asdict(cfg))
    small_cfg.budget = cfg.initial_size_budget
    small_cfg.strategies = ("Random", "Ensemble_UCB")

    for task_name in task_names(task_mode):
        true_score = observed_true_score(bundle.y, task_name, cfg)
        for initial_size in valid_sizes:
            small_cfg.initial_size = initial_size
            for repeat in range(1, cfg.initial_size_repeats + 1):
                initial = rng.choice(all_indices, size=initial_size, replace=False)
                candidates = np.setdiff1d(all_indices, initial)
                cache = RepeatPredictionCache()
                for strategy_index, strategy in enumerate(small_cfg.strategies):
                    traj, _ = run_one_strategy(
                        strategy=strategy,
                        bundle=bundle,
                        true_score=true_score,
                        task_name=task_name,
                        initial_indices=initial,
                        candidate_indices=candidates,
                        cfg=small_cfg,
                        device=device,
                        rng_seed=(
                                cfg.random_seed
                                + initial_size * 1009
                                + repeat * 10007
                                + strategy_index
                        ),
                        cache=cache,
                    )
                    final = traj.iloc[-1]
                    rows.append({
                        "task": task_name,
                        "initial_size": initial_size,
                        "strategy": strategy,
                        "repeat": repeat,
                        "best_score": float(final["best_score"]),
                        "selected_only_hit_rate_top10": float(
                            final["selected_only_hit_rate_top10"]
                        ),
                        "selected_only_hits_top10": int(
                            final["selected_only_hits_top10"]
                        ),
                        "hypervolume": float(final["hypervolume"]),
                    })
            print(f"Initial-size study complete: task={task_name}, n0={initial_size}")

    return pd.DataFrame(rows)


# =============================================================================
# PLOTS
# =============================================================================


def save_plot(fig_dir: Path, stem: str, cfg: BenchmarkCfg) -> None:
    plt.tight_layout()
    plt.savefig(fig_dir / f"{stem}.png", dpi=cfg.dpi)
    plt.savefig(fig_dir / f"{stem}.svg")
    plt.close()


def plot_best_so_far(traj_df: pd.DataFrame, fig_dir: Path, cfg: BenchmarkCfg) -> None:
    for task, sub in traj_df.groupby("task"):
        plt.figure(figsize=(8.8, 5.6))
        for strategy, group in sub.groupby("strategy"):
            stat = group.groupby("step")["best_score"].agg(
                ["mean", "std", "count"]
            ).reset_index()
            x = stat["step"].values
            mean = stat["mean"].values
            ci95 = 1.96 * stat["std"].fillna(0).values / np.sqrt(stat["count"].values)
            plt.plot(x, mean, marker="o", linewidth=2, label=strategy)
            plt.fill_between(x, mean - ci95, mean + ci95, alpha=0.15)
        plt.xlabel("Additional selected experiments")
        plt.ylabel("Best-so-far observed objective")
        plt.title(f"Paired retrospective search efficiency: {task}")
        plt.legend(fontsize=8)
        save_plot(fig_dir, f"fig1_best_so_far_{task}", cfg)


def plot_final_best(final_df: pd.DataFrame, fig_dir: Path, cfg: BenchmarkCfg) -> None:
    for task, sub in final_df.groupby("task"):
        order = list(
            sub.groupby("strategy")["best_score"].mean().sort_values(ascending=False).index
        )
        data = [sub.loc[sub["strategy"] == strategy, "best_score"].values for strategy in order]
        plt.figure(figsize=(9.0, 5.6))
        plt.boxplot(data, labels=order, showmeans=True)
        plt.ylabel("Final best observed objective")
        plt.xlabel("Selection strategy")
        plt.title(f"Final performance after a fixed budget: {task}")
        plt.xticks(rotation=28, ha="right")
        save_plot(fig_dir, f"fig2_final_best_{task}", cfg)


def plot_selected_hit_rate(final_df: pd.DataFrame, fig_dir: Path, cfg: BenchmarkCfg) -> None:
    metric = "selected_only_hit_rate_top10"
    for task, sub in final_df.groupby("task"):
        stat = sub.groupby("strategy")[metric].agg(["mean", "std"]).sort_values(
            "mean", ascending=False
        )
        plt.figure(figsize=(9.0, 5.6))
        plt.bar(stat.index, stat["mean"].values, yerr=stat["std"].values, capsize=4)
        plt.ylabel("Top-10% hit rate among newly selected experiments")
        plt.xlabel("Selection strategy")
        plt.title(f"High-performance candidate enrichment: {task}")
        plt.xticks(rotation=28, ha="right")
        save_plot(fig_dir, f"fig3_selected_hit_rate_{task}", cfg)


def plot_model_ablation(final_df: pd.DataFrame, fig_dir: Path, cfg: BenchmarkCfg) -> None:
    selected = [
        "CatBoost_greedy",
        "FPProcessNN_greedy",
        "Ensemble_greedy",
        "Ensemble_UCB",
    ]
    for task, sub in final_df.groupby("task"):
        sub = sub[sub["strategy"].isin(selected)]
        present = [strategy for strategy in selected if strategy in set(sub["strategy"])]
        data = [sub.loc[sub["strategy"] == strategy, "best_score"].values for strategy in present]
        plt.figure(figsize=(7.8, 5.3))
        plt.boxplot(data, labels=present, showmeans=True)
        plt.ylabel("Final best observed objective")
        plt.xlabel("Model/acquisition ablation")
        plt.title(f"Contribution of FPProcessNN, CatBoost, ensemble, and UCB: {task}")
        plt.xticks(rotation=25, ha="right")
        save_plot(fig_dir, f"fig4_model_ablation_{task}", cfg)


def plot_paired_gain(final_df: pd.DataFrame, fig_dir: Path, cfg: BenchmarkCfg) -> None:
    metric = "selected_only_hits_top10"
    for task, sub in final_df.groupby("task"):
        random_df = sub[sub["strategy"] == "Random"].set_index("repeat")
        strategies = [s for s in sub["strategy"].unique() if s != "Random"]
        gain_data: List[np.ndarray] = []
        labels: List[str] = []
        for strategy in strategies:
            model_df = sub[sub["strategy"] == strategy].set_index("repeat")
            common = random_df.index.intersection(model_df.index)
            gain = (
                    model_df.loc[common, metric].values
                    - random_df.loc[common, metric].values
            )
            gain_data.append(gain)
            labels.append(strategy)
        plt.figure(figsize=(9.0, 5.6))
        plt.boxplot(gain_data, labels=labels, showmeans=True)
        plt.axhline(0.0, linewidth=1)
        plt.ylabel("Paired gain in newly selected top-10% candidates vs Random")
        plt.xlabel("Selection strategy")
        plt.title(f"Repeat-wise paired improvement: {task}")
        plt.xticks(rotation=28, ha="right")
        save_plot(fig_dir, f"fig5_paired_gain_{task}", cfg)


def plot_initial_size(size_df: pd.DataFrame, fig_dir: Path, cfg: BenchmarkCfg) -> None:
    if size_df.empty:
        return
    metric = "selected_only_hit_rate_top10"
    for task, sub in size_df.groupby("task"):
        plt.figure(figsize=(7.8, 5.3))
        for strategy, group in sub.groupby("strategy"):
            stat = group.groupby("initial_size")[metric].agg(
                ["mean", "std", "count"]
            ).reset_index()
            ci95 = 1.96 * stat["std"].fillna(0).values / np.sqrt(stat["count"].values)
            x = stat["initial_size"].values
            mean = stat["mean"].values
            plt.plot(x, mean, marker="o", linewidth=2, label=strategy)
            plt.fill_between(x, mean - ci95, mean + ci95, alpha=0.15)
        plt.xlabel("Initial labeled experiments")
        plt.ylabel("Top-10% hit rate among newly selected experiments")
        plt.title(f"Initial-data-size sensitivity: {task}")
        plt.legend()
        save_plot(fig_dir, f"fig6_initial_size_{task}", cfg)


# =============================================================================
# REVIEWER REQUEST: STRICTLY PAIRED UCB - ENSEMBLE GREEDY ANALYSIS
# =============================================================================

def reviewer_paired_analysis(final_df: pd.DataFrame, data_dir: Path,
                             fig_dir: Path, cfg: BenchmarkCfg) -> pd.DataFrame:
    """Compare the *same ensemble* with beta=0 (greedy) versus beta=cfg.ucb_beta.

    Pairing unit: one task and one retrospective repeat, sharing the same
    initial labeled set and candidate pool within each task/repeat.
    Confidence intervals are descriptive percentile bootstrap CIs; n=3 per
    color is insufficient for a strong inferential claim. Pooled task/repeat
    rows should not be mistaken for independent experimental replicates.
    """
    from scipy.stats import binomtest

    metrics = (
        'selected_only_hit_rate_top10',
        'mean_score_newly_selected',
        'best_score',
        'hypervolume',
    )
    strategy_names = ('Ensemble_UCB', 'Ensemble_greedy')
    frame = final_df.loc[final_df['strategy'].isin(strategy_names)].copy()
    if frame.empty:
        raise ValueError('No Ensemble_UCB/Ensemble_greedy rows were found.')
    if frame.duplicated(['task', 'repeat', 'strategy']).any():
        raise ValueError('Duplicate (task, repeat, strategy) rows: pairing ambiguous.')
    keys = ['task', 'repeat']
    u = frame.loc[frame.strategy == 'Ensemble_UCB', keys + list(metrics)]
    g = frame.loc[frame.strategy == 'Ensemble_greedy', keys + list(metrics)]
    u = u.rename(columns={m: 'UCB_' + m for m in metrics})
    g = g.rename(columns={m: 'Greedy_' + m for m in metrics})
    pair = u.merge(g, on=keys, how='outer', indicator=True, validate='one_to_one')
    if not pair['_merge'].eq('both').all():
        bad = pair.loc[pair['_merge'] != 'both', keys + ['_merge']]
        raise ValueError(f'Unpaired rows exist; stop rather than silently drop:\n{bad}')
    pair = pair.drop(columns='_merge')
    for m in metrics:
        pair['Delta_' + m] = pair['UCB_' + m] - pair['Greedy_' + m]
    pair['UCB_hit_percent'] = 100 * pair['UCB_selected_only_hit_rate_top10']
    pair['Greedy_hit_percent'] = 100 * pair['Greedy_selected_only_hit_rate_top10']
    pair['Delta_hit_pp'] = 100 * pair['Delta_selected_only_hit_rate_top10']
    order = {'Blue': 0, 'Green': 1, 'Yellow': 2, 'Orange': 3}
    pair['_order'] = pair.task.map(order).fillna(999)
    pair = pair.sort_values(['_order', 'task', 'repeat']).drop(columns='_order')
    pair.to_csv(data_dir / 'reviewer_UCB_vs_greedy_all_paired_metrics.csv',
                index=False, encoding='utf-8-sig')
    pair[['task', 'repeat', 'UCB_hit_percent', 'Greedy_hit_percent', 'Delta_hit_pp']].to_csv(
        data_dir / 'reviewer_UCB_vs_greedy_hit_rate_pairs.csv',
        index=False, encoding='utf-8-sig')

    # Per-color estimates use repeats as pairing units. Percentile bootstrap
    # intervals and exact sign tests are descriptive with tiny sample sizes.
    rng = np.random.default_rng(2026)

    def stats(v, nboot=10000):
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if not len(v):
            return dict(n=0)
        boot = rng.choice(v, size=(nboot, len(v)), replace=True).mean(axis=1)
        nonzero = v[~np.isclose(v, 0, atol=1e-12, rtol=0)]
        sign_p = (float(binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5).pvalue)
                  if len(nonzero) else 1.0)
        try:
            wx_p = (float(wilcoxon(v, zero_method='wilcox', alternative='two-sided',
                                   method='auto').pvalue) if len(nonzero) else 1.0)
        except ValueError:
            wx_p = float('nan')
        return dict(n=len(v), mean=float(v.mean()), median=float(np.median(v)),
                    sd=float(v.std(ddof=1)) if len(v) > 1 else float('nan'),
                    ci95_lower=float(np.quantile(boot, .025)),
                    ci95_upper=float(np.quantile(boot, .975)),
                    ucb_better=int((v > 1e-12).sum()),
                    tied=int(np.isclose(v, 0, atol=1e-12, rtol=0).sum()),
                    ucb_worse=int((v < -1e-12).sum()),
                    wilcoxon_p=wx_p, exact_sign_p=sign_p)

    per_task = []
    overall = []
    for m in metrics:
        scale = 100 if m == 'selected_only_hit_rate_top10' else 1
        for task, group in pair.groupby('task', sort=False):
            item = stats(group['Delta_' + m].to_numpy() * scale)
            per_task.append(dict(task=task, metric=m,
                                 unit='percentage points' if scale == 100 else 'score units',
                                 mean_UCB=float(group['UCB_' + m].mean() * scale),
                                 mean_Greedy=float(group['Greedy_' + m].mean() * scale),
                                 **item))
        item = stats(pair['Delta_' + m].to_numpy() * scale)
        overall.append(dict(metric=m,
                            unit='percentage points' if scale == 100 else 'score units',
                            mean_UCB=float(pair['UCB_' + m].mean() * scale),
                            mean_Greedy=float(pair['Greedy_' + m].mean() * scale),
                            **item))
    pd.DataFrame(per_task).to_csv(data_dir / 'reviewer_UCB_vs_greedy_by_color.csv',
                                  index=False, encoding='utf-8-sig')
    pd.DataFrame(overall).to_csv(data_dir / 'reviewer_UCB_vs_greedy_pooled_descriptive.csv',
                                 index=False, encoding='utf-8-sig')

    # Paired raw-point graphic: each colored group has one point per repeat.
    tasks = list(dict.fromkeys(pair.task))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    jitter_width = .18
    for idx, task in enumerate(tasks):
        sub = pair[pair.task == task]
        x = np.full(len(sub), idx, dtype=float)
        jitter = np.linspace(-jitter_width, jitter_width, len(sub)) if len(sub) > 1 else np.zeros(1)
        axes[0].scatter(x + jitter, sub.Delta_hit_pp, s=52, edgecolors='white',
                        linewidths=.55, zorder=3)
        axes[0].plot([idx - .28, idx + .28], [sub.Delta_hit_pp.mean()] * 2,
                     color='black', lw=2.2, zorder=4)
        for i, row in enumerate(sub.itertuples()):
            axes[1].plot([idx - .14, idx + .14], [row.Greedy_hit_percent, row.UCB_hit_percent],
                         marker='o', markersize=4.5, lw=1.1, alpha=.75,
                         color='tab:blue' if row.Delta_hit_pp >= -1e-12 else 'tab:red')
    axes[0].axhline(0, color='gray', ls='--', lw=1)
    axes[0].set_ylabel('UCB − ensemble greedy (percentage points)')
    axes[0].set_title('(a) Paired hit-rate differences')
    axes[1].set_ylabel('Top-10% hit rate (%)')
    axes[1].set_title('(b) Greedy → UCB, paired repeats')
    for ax in axes:
        ax.set_xticks(range(len(tasks)), tasks)
        ax.set_xlabel('Target color')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.savefig(fig_dir / 'reviewer_paired_UCB_minus_greedy.png', dpi=cfg.dpi)
    fig.savefig(fig_dir / 'reviewer_paired_UCB_minus_greedy.pdf')
    plt.close(fig)

    # Optional raw-data table for Origin: separate wide-format paired values.
    wide = pair.pivot(index='repeat', columns='task', values='Delta_hit_pp')
    wide.to_csv(data_dir / 'reviewer_Origin_hit_rate_delta_wide.csv', encoding='utf-8-sig')
    print('\n=== REVIEWER REQUESTED PAIRED UCB − GREEDY TOP-10% HIT RATE ===')
    print(pair[['task', 'repeat', 'UCB_hit_percent', 'Greedy_hit_percent', 'Delta_hit_pp']]
          .to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print('\n=== PER-COLOR RESULTS (difference in percentage points) ===')
    summary = pd.DataFrame(per_task)
    print(summary.loc[summary.metric == 'selected_only_hit_rate_top10',
    ['task', 'n', 'mean', 'ci95_lower', 'ci95_upper', 'ucb_better', 'tied',
     'ucb_worse', 'wilcoxon_p']].to_string(index=False))
    print('\nNOTE: n=3/color produces weak statistical power. Pooled values are descriptive;')
    print('do not claim 4 colors × n repeats are wholly independent validation experiments.')
    return pair


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    global RUN_MODE, RUN_INITIAL_SIZE_STUDY
    import argparse
    parser = argparse.ArgumentParser(description='Full retrospective benchmark and reviewer paired ablation.')
    parser.add_argument('--run-mode', choices=['smoke', 'standard', 'publication'], default=RUN_MODE)
    parser.add_argument('--skip-initial-size-study', action='store_true')
    parser.add_argument('--analyze-only', metavar='CSV', default=None,
                        help='Skip all model training and analyze an existing benchmark_final_summary.csv')
    args = parser.parse_args()
    RUN_MODE = args.run_mode
    if args.skip_initial_size_study:
        RUN_INITIAL_SIZE_STUDY = False
    cfg = make_cfg(RUN_MODE)
    set_seed(cfg.random_seed, cfg.deterministic_torch)
    device = resolve_device(cfg)

    if not np.isclose(NN_FAMILY_WEIGHT + CB_FAMILY_WEIGHT, 1.0):
        raise ValueError("NN_FAMILY_WEIGHT + CB_FAMILY_WEIGHT must equal 1.")

    save_dir = Path(cfg.save_dir)
    data_dir = save_dir / "data"
    fig_dir = save_dir / "figures"
    ensure_dir(data_dir)
    ensure_dir(fig_dir)

    print("=" * 88)
    print("Exact Ensemble_mean_0p5NN_0p5CB retrospective benchmark")
    print(f"RUN_MODE={RUN_MODE}, TASK_MODE={TASK_MODE}, device={device}")
    print(f"seeds={cfg.seeds}")
    print(f"NN epochs={cfg.nn_epochs}, CatBoost iterations={cfg.cb_iterations}")
    print("=" * 88)

    if args.analyze_only:
        existing = pd.read_csv(args.analyze_only)
        reviewer_paired_analysis(existing, data_dir, fig_dir, cfg)
        print(f'Paired analysis saved to: {save_dir.resolve()}')
        return

    bundle = load_and_prepare_data(cfg)

    with open(data_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump({
            "run_mode": RUN_MODE,
            "task_mode": TASK_MODE,
            "experiment_file": EXPERIMENT_FILE,
            "smiles_path": SMILES_PATH,
            "feature_list_files": FEATURE_LIST_FILES,
            "color_windows": COLOR_WINDOWS,
            "ensemble_definition": "mean(concat(5 FPProcessNN, 5 CatBoost)) = 0.5NN + 0.5CB",
            "config": asdict(cfg),
        }, handle, ensure_ascii=False, indent=2)

    # Save the exact rows used, including observed targets and recipe metadata.
    bundle.df.to_csv(
        data_dir / "benchmark_dataset_used.csv",
        index=False,
        encoding="utf-8-sig",
    )

    trajectories, finals = run_benchmark(bundle, cfg, TASK_MODE, device)
    summary = make_strategy_summary(finals)
    tests = paired_tests(finals)

    trajectories.to_csv(
        data_dir / "benchmark_trajectory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    finals.to_csv(
        data_dir / "benchmark_final_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        data_dir / "benchmark_strategy_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    tests.to_csv(
        data_dir / "paired_statistical_tests.csv",
        index=False,
        encoding="utf-8-sig",
    )

    reviewer_paired_analysis(finals, data_dir, fig_dir, cfg)

    plot_best_so_far(trajectories, fig_dir, cfg)
    plot_final_best(finals, fig_dir, cfg)
    plot_selected_hit_rate(finals, fig_dir, cfg)
    plot_model_ablation(finals, fig_dir, cfg)
    plot_paired_gain(finals, fig_dir, cfg)

    if RUN_INITIAL_SIZE_STUDY:
        size_df = run_initial_size_study(bundle, cfg, TASK_MODE, device)
        size_df.to_csv(
            data_dir / "initial_size_study.csv",
            index=False,
            encoding="utf-8-sig",
        )
        plot_initial_size(size_df, fig_dir, cfg)

    print("\nStrategy summary:")
    print(summary.to_string(index=False))
    print("\nPaired statistical tests:")
    print(tests.to_string(index=False) if len(tests) else "No tests generated.")
    print(f"\nResults saved to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
