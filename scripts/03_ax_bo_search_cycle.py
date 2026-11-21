# 03_ax_bo_search_cycle.py
# -*- coding: utf-8 -*-
"""
Closed-loop Step 3:
Run Ax Bayesian optimization using the saved FPProcessNN + CatBoost ensemble.

This is true Ax search over the mixed synthesis space:
Ax proposes params -> saved ensemble predicts Ex/Em/QY -> objective returned to Ax.

Run:
    python 03_ax_bo_search_cycle.py

Before running:
    Set Cfg.run_dir to the same folder used in script 02.
"""
from __future__ import annotations
# Allow the documented "python scripts/..." entry point from a checkout.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, CatBoostClassifier

import torch
import torch.nn as nn
from ax.service.managed_loop import optimize

from importlib import import_module

training = import_module("scripts.02_train_fp_catboost_cycle")
build_compound_feature_table = training.build_compound_feature_table
build_dataset_features = training.build_dataset_features
from model.model import CarbonDotFeaturizer, data_process

import importlib.util


TARGET_NAMES = ["Ex_max_nm", "Em_max_nm", "QY_%"]

COLOR_WINDOWS = {
    "Blue": (430, 480),
    "Green": (500, 540),
    "Yellow": (560, 580),
    "Orange": (580, 630),
}

MOLECULAR_WEIGHTS = {
    "Salicylic acid": 138.12, "DL-Methionine": 149.21, "Glucose": 180.16,
    "2-Aminoterephthalic acid": 181.14, "L-Leucine": 131.17,
    "Dopamine hydrochloride": 189.64, "Urea": 60.06, "L-Phenylalanine": 165.19,
    "p-Aminobenzenesulfonic acid": 173.19, "Tartaric acid": 150.09,
    "p-Phenylenediamine": 108.14, "L-Cystine": 240.30, "L-Cysteine": 121.16,
    "m-Phenylenediamine": 108.14, "o-Phenylenediamine": 108.14,
    "Neutral Red": 288.78, "Citric acid": 192.12, "Glycine": 75.07,
    "L-Lysine": 146.19, "L-Tyrosine": 181.19, "Melamine": 126.12,
    "p-Aminophenol": 109.13, "L-Serine": 105.09, "L-Threonine": 119.12,
    "Ethanolamine": 61.08, "Terephthalic acid": 166.13, "Thiourea": 76.12,
}

REAGENT_LIST = ["water", "ethanol", "dimethylformamide", "sulfuric acid", "phosphoric acid"]

FEATURE_LIST_FILES = {
    "Ex_max_nm": "./cd_catboost_feature_selection/Ex_max_nm_best_top20_feature_list.csv",
    "Em_max_nm": "./cd_catboost_feature_selection/Em_max_nm_best_top35_feature_list.csv",
    "QY_%": "./cd_catboost_feature_selection/QY_%_best_top35_feature_list.csv",
}

FORBIDDEN_FEATURES = {"Ex_max_nm", "Em_max_nm", "QY_%", "stoke_shift", "stoke_shift_calc_check", "Stokes_shift_nm"}


@dataclass
class Cfg:
    cycle: int = 1
    run_dir: str = "runs/cycle_01"
    smiles_path: str = "./data/compound_smiles.xlsx"
    observed_history_file: str = "./data/observed_history.xlsx"


    precursor_mass: tuple = (0.2, 0.4, 0.6, 0.8, 1.0)
    temperature_values: tuple = (160,180,200,220)
    time_values: tuple = (4,6,8)
    allow_same_precursor: bool = True
    reagent_1_mL: float = 15.0

    total_trials: int = 200
    runs_per_color: int = 5
    batch_select_per_color: int = 1

    alpha: float = 0.7
    color_softness_nm: float = 10

    qy_min_percent: float = 0.0
    qy_max_percent: float = 100.0
    stokes_min_nm: float = 40.0
    stokes_max_nm: float = 180.0

    # mean_score or ucb_score
    objective_mode: str = "ucb_score"
    ucb_beta: float = 0.2

    # Success probability gate:
    # final_objective = P_success * (score_mu + beta * score_sigma)
    use_success_probability: bool = True
    use_success_hard_gate: bool = True
    success_min_probability: float = 0.50
    success_feature_file_name: str = "success_classifier_feature_list.csv"

    # Exclude recipes that have already appeared in the expanded training data
    # and/or observed_history. Recommended for prospective discovery.
    exclude_tested_recipes: bool = True
    duplicate_key_mode: str = "unordered_pair"  # "exact_order" or "unordered_pair"
    recipe_key_decimals: int = 4

    use_offline_stopping: bool = True
    min_cycles: int = 3
    max_cycles: int = 16
    patience_cycles: int = 4
    abs_tolerance: float = 0.02
    rel_tolerance: float = 0.03
    score_noise_floor: float = 0.02

    random_seed: int = 42
    dpi: int = 300


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def read_table(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_excel(p) if p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(p)


def minmax_fixed(x, lo, hi):
    return 0.0 if hi <= lo else float(np.clip((x-lo)/(hi-lo), 0.0, 1.0))


def color_fit_scalar(em, lower, upper, softness):
    center, half = 0.5*(lower+upper), 0.5*(upper-lower)
    outside = max(abs(float(em)-center)-half, 0.0)
    return float(np.exp(-(outside**2)/(2.0*softness**2)))


def color_fit_array(em_arr, lower, upper, softness):
    em = np.asarray(em_arr, dtype=float)
    center, half = 0.5*(lower+upper), 0.5*(upper-lower)
    outside = np.maximum(np.abs(em-center)-half, 0.0)
    return np.exp(-(outside**2)/(2.0*softness**2))


def mass_to_mmol(mass_g, mw):
    return float(mass_g)/float(mw)*1000.0


def _get_mass_from_row(row, precursor_col, mass_col, mass_alt_col, mmol_col):
    """
    Return precursor mass in g. Prefer mass column; if absent, back-calculate
    from mmol using molecular weight.
    """
    if mass_col in row.index and pd.notna(row.get(mass_col)):
        return float(row.get(mass_col))
    if mass_alt_col in row.index and pd.notna(row.get(mass_alt_col)):
        return float(row.get(mass_alt_col))
    if mmol_col in row.index and pd.notna(row.get(mmol_col)):
        precursor = str(row.get(precursor_col))
        if precursor in MOLECULAR_WEIGHTS:
            return float(row.get(mmol_col)) * MOLECULAR_WEIGHTS[precursor] / 1000.0
    return np.nan


def recipe_key_from_values(p1, p2, mass1, mass2, temperature, time_h, reagent, cfg):
    """
    Build a recipe key for excluding already tested recipes.

    In unordered_pair mode:
        precursor_1=A, mass_1=x, precursor_2=B, mass_2=y
    and
        precursor_1=B, mass_1=y, precursor_2=A, mass_2=x
    are treated as the same experimental recipe.
    """
    p1 = str(p1).strip()
    p2 = str(p2).strip()
    reagent = str(reagent).strip()
    m1 = round(float(mass1), cfg.recipe_key_decimals)
    m2 = round(float(mass2), cfg.recipe_key_decimals)
    temp = int(temperature)
    th = int(time_h)

    if cfg.duplicate_key_mode == "unordered_pair":
        pair = sorted([(p1, m1), (p2, m2)], key=lambda x: (x[0], x[1]))
        return (pair[0][0], pair[0][1], pair[1][0], pair[1][1], reagent, temp, th)

    if cfg.duplicate_key_mode == "exact_order":
        return (p1, m1, p2, m2, reagent, temp, th)

    raise ValueError("duplicate_key_mode must be 'unordered_pair' or 'exact_order'.")


def recipe_key_from_params(params, cfg):
    return recipe_key_from_values(
        params["precursor_1"], params["precursor_2"],
        params["mass_1_g"], params["mass_2_g"],
        params["temperature"], params["time_h"], params["reagent_1"], cfg
    )


def recipe_key_from_row(row, cfg):
    p1 = row.get("precursor_1")
    p2 = row.get("precursor_2")
    reagent = row.get("reagent_1", row.get("reagent"))
    mass1 = _get_mass_from_row(row, "precursor_1", "mass_1_g", "Mass1(g)", "pre_1_mmol")
    mass2 = _get_mass_from_row(row, "precursor_2", "mass_2_g", "Mass2(g)", "pre_2_mmol")

    if pd.isna(mass1) or pd.isna(mass2):
        return None

    return recipe_key_from_values(
        p1, p2, mass1, mass2,
        row.get("temperature"), row.get("time_h"), reagent, cfg
    )


def load_tested_recipe_keys(cfg):
    """
    Load recipe keys from observed_history and expanded training data.
    These recipes are excluded from future prospective BO recommendations.
    """
    keys = set()
    source_counts = {}

    candidate_files = []

    if cfg.observed_history_file:
        candidate_files.append(("observed_history", Path(cfg.observed_history_file)))

    expanded_xlsx = Path(cfg.run_dir) / "data" / f"expanded_training_data_cycle_{cfg.cycle:02d}.xlsx"
    expanded_csv = Path(cfg.run_dir) / "data" / f"expanded_training_data_cycle_{cfg.cycle:02d}.csv"
    candidate_files.append(("expanded_training_xlsx", expanded_xlsx))
    candidate_files.append(("expanded_training_csv", expanded_csv))

    for source_name, path in candidate_files:
        if not path.exists():
            continue

        try:
            df = read_table(str(path))
            if source_name == "observed_history":
                df = df.loc[pd.to_numeric(df["cycle"], errors="raise") < cfg.cycle]
        except Exception as e:
            print(f"[WARNING] Failed to read tested recipe source {path}: {e}")
            continue

        if "reagent" in df.columns and "reagent_1" not in df.columns:
            df = df.rename(columns={"reagent": "reagent_1"})

        required_basic = {"precursor_1", "precursor_2", "reagent_1", "temperature", "time_h"}
        if not required_basic.issubset(df.columns):
            continue

        n_added = 0
        for _, row in df.iterrows():
            try:
                key = recipe_key_from_row(row, cfg)
                if key is not None:
                    keys.add(key)
                    n_added += 1
            except Exception:
                continue

        source_counts[source_name] = n_added

    print(f"[INFO] Tested-recipe exclusion enabled. Loaded {len(keys)} unique tested recipe keys.")
    print(f"[INFO] Tested-recipe source counts: {source_counts}")
    return keys



def get_solvent_polarity_data():
    return {
        "water": {"dielectric_constant":80.1,"pi_star":1.09,"HBD":1.17,"HBA":0.47,"ET30":63.1,"pKa":15.7,"ORP_mV":300,"density_g_cm3":0.997,"specific_heat_J_gK":4.18,"viscosity_mPa_s":0.89,"boiling_point_C":100},
        "dimethylformamide": {"dielectric_constant":36.7,"pi_star":0.88,"HBD":0.00,"HBA":0.69,"ET30":43.2,"pKa":29,"ORP_mV":200,"density_g_cm3":0.944,"specific_heat_J_gK":2.47,"viscosity_mPa_s":0.92,"boiling_point_C":153},
        "ethanol": {"dielectric_constant":24.6,"pi_star":0.54,"HBD":0.86,"HBA":0.75,"ET30":51.9,"pKa":15.9,"ORP_mV":250,"density_g_cm3":0.789,"specific_heat_J_gK":2.44,"viscosity_mPa_s":1.07,"boiling_point_C":78.37},
        "sulfuric acid": {"dielectric_constant":83,"pi_star":1.11,"HBD":1.22,"HBA":0.48,"ET30":64.8,"pKa":0.0,"ORP_mV":800,"density_g_cm3":1.05,"specific_heat_J_gK":3.6,"viscosity_mPa_s":1.2,"boiling_point_C":100},
        "phosphoric acid": {"dielectric_constant":82,"pi_star":1.10,"HBD":1.20,"HBA":0.47,"ET30":64.0,"pKa":1.0,"ORP_mV":700,"density_g_cm3":1.03,"specific_heat_J_gK":3.8,"viscosity_mPa_s":1.1,"boiling_point_C":100},
    }


class FingerprintEncoderSmall(nn.Module):
    def __init__(self, input_dim=679, latent_dim=32, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim,128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128,latent_dim), nn.ReLU())
    def forward(self, x): return self.net(x.float())


class ProcessEncoderSmall(nn.Module):
    def __init__(self, input_dim=14, out_dim=16, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim,out_dim), nn.ReLU(), nn.Dropout(dropout))
    def forward(self, x): return self.net(x.float())


class FPProcessNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp_encoder = FingerprintEncoderSmall()
        self.proc_encoder = ProcessEncoderSmall()
        self.fusion_mlp = nn.Sequential(nn.Linear(64,64), nn.GELU(), nn.Dropout(0.30), nn.Linear(64,64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(80,64), nn.GELU(), nn.Dropout(0.30), nn.Linear(64,3))
    def forward(self, fps_1, fps_2, global_feat):
        f1, f2 = self.fp_encoder(fps_1), self.fp_encoder(fps_2)
        fusion = self.fusion_mlp(torch.cat([f1+f2, torch.abs(f1-f2)], dim=-1))
        proc = self.proc_encoder(global_feat)
        return self.head(torch.cat([fusion, proc], dim=-1))


def read_feature_list(path):
    df = read_table(path)
    if "feature" not in df.columns:
        raise ValueError(f"Feature-list file must contain feature column: {path}")
    features = list(dict.fromkeys(df["feature"].dropna().astype(str).str.strip().tolist()))
    leak = [f for f in features if f in FORBIDDEN_FEATURES]
    if leak:
        raise ValueError(f"Feature list contains leakage columns: {leak}")
    return features


class EnsemblePredictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.model_dir = self.run_dir/"models"
        self.data_dir = self.run_dir/"data"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        scaler_path = self.model_dir/"y_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.y_scaler = pickle.load(f)
        else:
            train_path = self.data_dir/f"expanded_training_data_cycle_{cfg.cycle:02d}.xlsx"
            train_df = data_process(read_table(str(train_path)))
            self.y_scaler = StandardScaler().fit(train_df[TARGET_NAMES].astype(float).values)

        smiles = pd.read_excel(cfg.smiles_path)
        self.featurizer = CarbonDotFeaturizer(smiles, get_solvent_polarity_data())

        smiles_compound = pd.read_excel(cfg.smiles_path, sheet_name="compound")
        self.compound_feat_df, self.fp_objs = build_compound_feature_table(smiles_compound)

        self.feature_lists = {t: read_feature_list(FEATURE_LIST_FILES[t]) for t in TARGET_NAMES}
        self.success_features = self.load_success_feature_list()
        self.nn_models = self.load_nn()
        self.cb_models = self.load_cb()
        self.success_models = self.load_success_models()
        self.cache = {}
        self.tested_recipe_keys = load_tested_recipe_keys(cfg) if cfg.exclude_tested_recipes else set()

    def load_success_feature_list(self):
        path = self.data_dir / self.cfg.success_feature_file_name
        if not path.exists():
            print(f"[WARNING] Success feature file not found: {path}. P_success will be set to 1.")
            return []
        df = pd.read_csv(path)
        if "feature" not in df.columns:
            raise ValueError(f"Success feature file must contain feature column: {path}")
        return list(dict.fromkeys(df["feature"].dropna().astype(str).str.strip().tolist()))

    def load_success_models(self):
        paths = sorted(self.model_dir.glob("catboost_success_seed*.cbm"))
        if not paths:
            print("[WARNING] No catboost_success_seed*.cbm found. P_success will be set to 1.")
            return []
        models = []
        for p in paths:
            m = CatBoostClassifier()
            m.load_model(str(p))
            models.append(m)
        print(f"Loaded CatBoost success classifiers: {len(models)}")
        return models

    def load_nn(self):
        paths = sorted(self.model_dir.glob("FPProcessNN_seed*.pt"))
        if not paths:
            raise FileNotFoundError(f"No FPProcessNN_seed*.pt found in {self.model_dir}")
        models = []
        for p in paths:
            m = FPProcessNN().to(self.device)
            m.load_state_dict(torch.load(p, map_location=self.device))
            m.eval()
            models.append(m)
        print(f"Loaded FPProcessNN models: {len(models)}")
        return models

    def load_cb(self):
        out = {}
        for target in TARGET_NAMES:
            paths = sorted(self.model_dir.glob(f"catboost_{target}_seed*.cbm"))
            if not paths:
                raise FileNotFoundError(f"No CatBoost model for {target} in {self.model_dir}")
            models = []
            for p in paths:
                m = CatBoostRegressor(); m.load_model(str(p)); models.append(m)
            out[target] = models
            print(f"Loaded CatBoost {target}: {len(models)}")
        return out

    def params_to_df(self, params):
        p1, p2 = str(params["precursor_1"]), str(params["precursor_2"])
        mass1, mass2 = float(params["mass_1_g"]), float(params["mass_2_g"])
        pre1, pre2 = mass_to_mmol(mass1, MOLECULAR_WEIGHTS[p1]), mass_to_mmol(mass2, MOLECULAR_WEIGHTS[p2])
        return pd.DataFrame([{
            "candidate_id": -1, "precursor_1": p1, "precursor_2": p2,
            "reagent_1": str(params["reagent_1"]),
            "mass_1_g": mass1, "mass_2_g": mass2, "Mass1(g)": mass1, "Mass2(g)": mass2,
            "pre_1_mmol": pre1, "pre_2_mmol": pre2, "reagent_1_mL": self.cfg.reagent_1_mL,
            "temperature": int(params["temperature"]), "time_h": int(params["time_h"]),
        }])

    def key(self, params):
        return (
            str(params["precursor_1"]), str(params["precursor_2"]),
            round(float(params["mass_1_g"]), 6), round(float(params["mass_2_g"]), 6),
            int(params["temperature"]), int(params["time_h"]), str(params["reagent_1"])
        )

    def predict_success(self, feat_df: pd.DataFrame) -> float:
        """
        Predict P_success using the CatBoostClassifier ensemble.
        If no success model is available, return 1.0 for backward compatibility.
        """
        if not self.success_models or not self.success_features:
            return 1.0

        missing = [c for c in self.success_features if c not in feat_df.columns]
        if missing:
            raise ValueError(f"Missing success-classifier features: {missing[:30]}")

        X = feat_df[self.success_features].copy()
        for c in self.success_features:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

        probs = []
        for m in self.success_models:
            p = m.predict_proba(X)[0, 1]
            probs.append(float(p))
        return float(np.mean(probs))

    def predict(self, params):
        k = self.key(params)
        if k in self.cache:
            return self.cache[k].copy()

        df = self.params_to_df(params)
        p1, p2, reagent = df.loc[0, "precursor_1"], df.loc[0, "precursor_2"], df.loc[0, "reagent_1"]

        fp1 = torch.tensor(self.featurizer.fingerprints(p1), dtype=torch.float32).unsqueeze(0).to(self.device)
        fp2 = torch.tensor(self.featurizer.fingerprints(p2), dtype=torch.float32).unsqueeze(0).to(self.device)
        s_feat = self.featurizer.solvent_feat(reagent).float()
        r_feat = self.featurizer.reaction_feat(df.loc[0, "temperature"], df.loc[0, "time_h"]).float()
        pre1, pre2 = float(df.loc[0, "pre_1_mmol"]), float(df.loc[0, "pre_2_mmol"])
        ratio = pre1/(pre1+pre2) if (pre1+pre2)>0 else 0.5
        global_feat = torch.cat([s_feat, r_feat, torch.tensor([ratio], dtype=torch.float32)]).unsqueeze(0).to(self.device)

        nn_preds = []
        with torch.no_grad():
            for m in self.nn_models:
                y_scaled = m(fp1, fp2, global_feat).detach().cpu().numpy()
                nn_preds.append(self.y_scaler.inverse_transform(y_scaled)[0].astype(np.float32))
        nn_preds = np.stack(nn_preds, axis=0)

        feat_df = build_dataset_features(df, self.compound_feat_df, self.fp_objs, has_targets=False)
        p_success = self.predict_success(feat_df)
        
        cb_targets = []
        for target in TARGET_NAMES:
            features = self.feature_lists[target]
            missing = [c for c in features if c not in feat_df.columns]
            if missing:
                raise ValueError(f"Missing CatBoost features for {target}: {missing[:30]}")
            X = feat_df[features].copy()
            for c in features:
                X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
            preds = [float(m.predict(X)[0]) for m in self.cb_models[target]]
            cb_targets.append(np.array(preds, dtype=np.float32))
        cb_preds = np.stack(cb_targets, axis=-1)

        members = np.concatenate([nn_preds, cb_preds], axis=0)
        mean, std = members.mean(axis=0), members.std(axis=0)
        ex, em, qy = mean.tolist()
        ex_std, em_std, qy_std = std.tolist()

        out = df.iloc[0].to_dict()
        out.update({
            "Ensemble_pred_Ex_max_nm": float(ex),
            "Ensemble_pred_Em_max_nm": float(em),
            "Ensemble_pred_QY_%": float(qy),
            "Ensemble_pred_Stokes_shift_nm": float(em-ex),
            "Ensemble_std_Ex_max_nm": float(ex_std),
            "Ensemble_std_Em_max_nm": float(em_std),
            "Ensemble_std_QY_%": float(qy_std),
            "Ensemble_std_Stokes_shift_nm": float(np.sqrt(ex_std**2 + em_std**2)),
            "P_success": float(p_success),
        })
        self.cache[k] = out
        return out

    def score(self, params, color):
        pred = self.predict(params)
        lower, upper = COLOR_WINDOWS[color]
        fit = color_fit_scalar(pred["Ensemble_pred_Em_max_nm"], lower, upper, self.cfg.color_softness_nm)
        qy_s = minmax_fixed(pred["Ensemble_pred_QY_%"], self.cfg.qy_min_percent, self.cfg.qy_max_percent)
        stokes_s = minmax_fixed(pred["Ensemble_pred_Stokes_shift_nm"], self.cfg.stokes_min_nm, self.cfg.stokes_max_nm)
        score_mu = fit * (self.cfg.alpha*qy_s + (1-self.cfg.alpha)*stokes_s)

        qy_std_norm = pred["Ensemble_std_QY_%"] / max(self.cfg.qy_max_percent - self.cfg.qy_min_percent, 1e-8)
        stokes_std_norm = pred["Ensemble_std_Stokes_shift_nm"] / max(self.cfg.stokes_max_nm - self.cfg.stokes_min_nm, 1e-8)
        score_sigma = fit * (self.cfg.alpha*qy_std_norm + (1-self.cfg.alpha)*stokes_std_norm)

        if self.cfg.objective_mode == "mean_score":
            base_objective = score_mu
        elif self.cfg.objective_mode == "ucb_score":
            base_objective = score_mu + self.cfg.ucb_beta * score_sigma
        else:
            raise ValueError("objective_mode must be 'mean_score' or 'ucb_score'.")

        p_success = float(pred.get("P_success", 1.0))

        if self.cfg.use_success_probability:
            objective = p_success * base_objective
        else:
            objective = base_objective

        if self.cfg.use_success_hard_gate and p_success < self.cfg.success_min_probability:
            objective = 0.0

        pred.update({
            "target_color": color,
            "color_fit": fit,
            "QY_score_fixed": qy_s,
            "Stokes_score_fixed": stokes_s,
            "score_mu": float(score_mu),
            "score_sigma_approx": float(score_sigma),
            "base_objective_before_success": float(base_objective),
            "P_success": float(p_success),
            "objective": float(objective),
        })
        return pred


def load_history(cfg):
    p = Path(cfg.observed_history_file)
    if not p.exists(): return None
    h = read_table(str(p))
    h = h.loc[pd.to_numeric(h["cycle"], errors="raise") < cfg.cycle].copy()
    if "reagent" in h.columns and "reagent_1" not in h.columns:
        h = h.rename(columns={"reagent": "reagent_1"})
    req = ["cycle","precursor_1","precursor_2","reagent_1","temperature","time_h","Ex_max_nm","Em_max_nm","QY_%"]
    missing = [c for c in req if c not in h.columns]
    if missing: raise ValueError(f"observed_history missing columns: {missing}")
    if "Stokes_shift_nm" not in h.columns:
        h["Stokes_shift_nm"] = h["Em_max_nm"] - h["Ex_max_nm"]
    return h


def evaluate_stopping(history, cfg):
    if history is None or len(history) == 0:
        return pd.DataFrame([
            {
                "target_color": c,
                "stop": False,
                "continue_recommendation": True,
                "reason": "no observed history"
            }
            for c in COLOR_WINDOWS
        ]), None

    h = history.copy()
    h["QY_score_fixed_obs"] = h["QY_%"].apply(
        lambda v: minmax_fixed(v, cfg.qy_min_percent, cfg.qy_max_percent)
    )
    h["Stokes_score_fixed_obs"] = h["Stokes_shift_nm"].apply(
        lambda v: minmax_fixed(v, cfg.stokes_min_nm, cfg.stokes_max_nm)
    )

    rows = []

    for color, (lower, upper) in COLOR_WINDOWS.items():

        # 建议只使用该颜色目标对应的历史实验
        h_color = h[h["target_color"].astype(str).str.lower() == color.lower()].copy()

        if len(h_color) == 0:
            rows.append({
                "target_color": color,
                "stop": False,
                "continue_recommendation": True,
                "reason": "no observed samples for this target color",
                "n_cycles_completed": 0,
                "current_best_observed_score": 0.0,
                "consecutive_no_significant_improvement": 0
            })
            continue

        # 判断是否已经有实验结果真正落入目标颜色窗口
        h_color["is_color_hit"] = h_color["Em_max_nm"].between(lower, upper)

        if not h_color["is_color_hit"].any():
            rows.append({
                "target_color": color,
                "stop": False,
                "continue_recommendation": True,
                "reason": "no experimentally validated sample within target color window",
                "n_cycles_completed": int(h_color["cycle"].max()),
                "current_best_observed_score": 0.0,
                "consecutive_no_significant_improvement": 0
            })
            continue

        fit = color_fit_array(
            h_color["Em_max_nm"].values,
            lower,
            upper,
            cfg.color_softness_nm
        )

        h_color[f"{color}_score_obs"] = fit * (
            cfg.alpha * h_color["QY_score_fixed_obs"].values
            + (1 - cfg.alpha) * h_color["Stokes_score_fixed_obs"].values
        )

        cyc = (
            h_color.groupby("cycle")[f"{color}_score_obs"]
            .max()
            .reset_index(name="cycle_best")
            .sort_values("cycle")
        )

        cyc["best_so_far"] = cyc["cycle_best"].cummax()
        cyc["prev_best"] = cyc["best_so_far"].shift(1)
        cyc["improvement"] = cyc["best_so_far"] - cyc["prev_best"]
        cyc.loc[cyc["prev_best"].isna(), "improvement"] = np.nan

        cyc["tolerance"] = np.maximum.reduce([
            np.full(len(cyc), cfg.abs_tolerance),
            cfg.rel_tolerance * cyc["prev_best"].fillna(0.0).values,
            np.full(len(cyc), cfg.score_noise_floor),
        ])

        cyc["significant_improvement"] = cyc["improvement"] > cyc["tolerance"]

        n_cycles = int(cyc["cycle"].max())

        no_improve = 0
        for flag in cyc.dropna(subset=["improvement"])["significant_improvement"].iloc[::-1]:
            if bool(flag):
                break
            no_improve += 1

        stop, reasons = False, []

        if n_cycles >= cfg.max_cycles:
            stop = True
            reasons.append(f"maximum cycle budget reached: {cfg.max_cycles}")

        if n_cycles >= cfg.min_cycles and no_improve >= cfg.patience_cycles:
            stop = True
            reasons.append(
                f"no significant observed improvement for {no_improve} consecutive cycles after color hit"
            )

        if not reasons:
            reasons.append("continue")

        rows.append({
            "target_color": color,
            "stop": stop,
            "continue_recommendation": not stop,
            "reason": "; ".join(reasons),
            "n_cycles_completed": n_cycles,
            "current_best_observed_score": float(cyc["best_so_far"].max()),
            "consecutive_no_significant_improvement": no_improve
        })

    return pd.DataFrame(rows), h


def ax_parameters(cfg):
    precursors = list(MOLECULAR_WEIGHTS.keys())
    return [
        {"name":"precursor_1","type":"choice","values":precursors,"value_type":"str"},
        {"name":"precursor_2","type":"choice","values":precursors,"value_type":"str"},
        {"name":"mass_1_g","type":"choice","values":list(cfg.precursor_mass),"value_type":"float"},
        {"name":"mass_2_g","type":"choice","values":list(cfg.precursor_mass),"value_type":"float"},
        {"name":"temperature","type":"choice","values":list(cfg.temperature_values),"value_type":"int"},
        {"name":"time_h","type":"choice","values":list(cfg.time_values),"value_type":"int"},
        {"name":"reagent_1","type":"choice","values":REAGENT_LIST,"value_type":"str"},
    ]


def run_bo_color(color, predictor, cfg):
    params_def = ax_parameters(cfg)
    evals, bests, trials = [], [], []
    def objective(params):
        if (not cfg.allow_same_precursor) and params["precursor_1"] == params["precursor_2"]:
            rec = dict(params)
            rec.update({"target_color": color, "objective": 0.0, "exclusion_reason": "same_precursor_not_allowed"})
            evals.append(rec)
            return 0.0

        if cfg.exclude_tested_recipes:
            try:
                key = recipe_key_from_params(params, cfg)
                if key in predictor.tested_recipe_keys:
                    rec = dict(params)
                    rec.update({"target_color": color, "objective": 0.0, "exclusion_reason": "tested_recipe"})
                    evals.append(rec)
                    return 0.0
            except Exception as e:
                rec = dict(params)
                rec.update({"target_color": color, "objective": 0.0, "error": f"recipe_key_error: {e}"})
                evals.append(rec)
                return 0.0

        try:
            rec = predictor.score(params, color)
            rec["exclusion_reason"] = ""
            evals.append(rec)
            return float(rec["objective"])
        except Exception as e:
            err = dict(params); err.update({"target_color":color,"objective":0.0,"error":str(e)})
            evals.append(err)
            return 0.0

    for run in range(1, cfg.runs_per_color+1):
        print(f"\n[Ax BO] color={color}, run={run}/{cfg.runs_per_color}")
        best_params, values, experiment, model = optimize(
            parameters=params_def, evaluation_function=objective,
            objective_name="objective", total_trials=cfg.total_trials, minimize=False
        )
        best = predictor.score(best_params, color); best["ax_run"] = run; best["selection_route"] = "best_per_ax_run"
        bests.append(best)
        try:
            tr = experiment.fetch_data().df.copy(); tr["target_color"] = color; tr["ax_run"] = run
            trials.append(tr)
        except Exception:
            pass
    return pd.DataFrame(bests), pd.DataFrame(evals), pd.concat(trials, ignore_index=True) if trials else pd.DataFrame()


def select_diverse(eval_df, cfg):
    if len(eval_df)==0: return pd.DataFrame()
    valid = eval_df.copy()
    if "error" in valid.columns:
        valid = valid[valid["error"].isna()].copy()
    if "exclusion_reason" in valid.columns:
        valid = valid[(valid["exclusion_reason"].isna()) | (valid["exclusion_reason"].astype(str) == "")].copy()
    valid = valid[valid["objective"] > 0].copy()
    out_all = []
    for color, sub in valid.groupby("target_color"):
        sub = sub.sort_values("objective", ascending=False)
        selected, used_pairs = [], set()
        for _, r in sub.iterrows():
            pair = tuple(sorted([str(r["precursor_1"]), str(r["precursor_2"])]))
            if pair not in used_pairs:
                selected.append(r); used_pairs.add(pair)
            if len(selected) >= cfg.batch_select_per_color: break
        if len(selected) < cfg.batch_select_per_color:
            for _, r in sub.iterrows():
                selected.append(r)
                if len(selected) >= cfg.batch_select_per_color: break
        temp = pd.DataFrame(selected).reset_index(drop=True)
        temp.insert(0, "selected_order", np.arange(1, len(temp)+1))
        out_all.append(temp)
    return pd.concat(out_all, ignore_index=True) if out_all else pd.DataFrame()


def plot_best(eval_df, fig_dir, cfg):
    if len(eval_df)==0: return
    plt.figure(figsize=(8,5))
    for color, sub in eval_df.groupby("target_color"):
        sub = sub.reset_index(drop=True)
        sub["best_so_far"] = sub["objective"].cummax()
        plt.plot(np.arange(1, len(sub)+1), sub["best_so_far"], label=color)
    plt.xlabel("Ax objective evaluations")
    plt.ylabel("Best-so-far objective")
    plt.title("Ax BO live-model search trajectory")
    plt.legend(); plt.tight_layout()
    plt.savefig(fig_dir/"ax_bo_best_objective_by_trial.png", dpi=cfg.dpi)
    plt.close()


def main(cfg=None):
    cfg = cfg or Cfg()
    set_seed(cfg.random_seed)
    run_dir = Path(cfg.run_dir)
    data_dir, fig_dir = run_dir/"data", run_dir/"figures"
    data_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir/"ax_bo_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    history = load_history(cfg) if cfg.use_offline_stopping else None
    stopping, history_scored = evaluate_stopping(history, cfg)
    stopping.to_csv(data_dir/"stopping_status_by_color.csv", index=False, encoding="utf-8-sig")
    if history_scored is not None:
        history_scored.to_csv(data_dir/"observed_history_with_scores.csv", index=False, encoding="utf-8-sig")

    print("\nOffline stopping status:")
    print(stopping.to_string(index=False))
    active = stopping.loc[stopping["continue_recommendation"], "target_color"].tolist()
    if not active:
        print("All colors stopped. BO recommendation skipped.")
        return

    predictor = EnsemblePredictor(cfg)

    best_all, eval_all, trial_all = [], [], []
    for color in active:
        b, e, t = run_bo_color(color, predictor, cfg)
        best_all.append(b); eval_all.append(e)
        if len(t): trial_all.append(t)

    best_all = pd.concat(best_all, ignore_index=True) if best_all else pd.DataFrame()
    eval_all = pd.concat(eval_all, ignore_index=True) if eval_all else pd.DataFrame()
    trial_all = pd.concat(trial_all, ignore_index=True) if trial_all else pd.DataFrame()
    selected = select_diverse(eval_all, cfg)

    best_all.to_csv(data_dir/"ax_bo_best_per_run_by_color.csv", index=False, encoding="utf-8-sig")
    eval_all.to_csv(data_dir/"ax_bo_all_evaluations_by_color.csv", index=False, encoding="utf-8-sig")
    trial_all.to_csv(data_dir/"ax_bo_trials_raw_by_color.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(data_dir/"ax_bo_recommendations_by_color.csv", index=False, encoding="utf-8-sig")
    selected.to_excel(data_dir/"ax_bo_recommendations_by_color.xlsx", index=False)
    plot_best(eval_all, fig_dir, cfg)

    show= [ "target_color", "selected_order","precursor_1", "precursor_2",
    "mass_1_g", "mass_2_g","pre_1_mmol", "pre_2_mmol","temperature", "time_h", "reagent_1",
    "Ensemble_pred_Ex_max_nm","Ensemble_pred_Em_max_nm","Ensemble_pred_QY_%","Ensemble_pred_Stokes_shift_nm",
    "color_fit","P_success","score_mu","score_sigma_approx","base_objective_before_success","objective","exclusion_reason",]
    show = [c for c in show if c in selected.columns]
    print("\nFinal selected BO recommendations:")
    print(selected[show].to_string(index=False))
    print(f"\nResults saved in: {run_dir}")


if __name__ == "__main__":
    from workflow.stages import stage_cli
    stage_cli(3)
