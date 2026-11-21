#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark property-prediction models on success=1 experiments only.

Important design choices
------------------------
1. The initial Excel table is filtered first: only rows with success == 1 remain.
2. No pre-generated CatBoost feature table is read.
3. All RDKit, fingerprint, precursor-pair, solvent, process and QY-oriented
   feature-engineering functions are embedded directly in this file.
4. FEATURE_LIST_FILES are used only to select target-specific columns.
5. Identical five model configurations are compared under random, grouped, and leave-one-pair-out splits.
6. CatBoost feature generation is performed once for the filtered rows, while
   every model fit remains strictly fold-specific.
7. Preselected target-specific feature lists are fixed: this is a split-sensitivity
   comparison, NOT a leak-free nested model selection assessment.

Models compared (same architecture as 02_train_fp_catboost_cycle.py)
-------------------------------------------------------------------
- FPNN (679-128-32 fingerprint encoder; 14-16 process encoder)
- target-specific CatBoost regressors
- 0.3FPNN + 0.7CatBoost
- 0.5FPNN + 0.5CatBoost (production ensemble)
- 0.7FPNN + 0.3CatBoost

Required files (default paths)
------------------------------
./data/output_with_molar_amount_pubchem.xlsx
./data/compound_smiles.xlsx
./cd_catboost_feature_selection/Ex_max_nm_best_top20_feature_list.csv
./cd_catboost_feature_selection/Em_max_nm_best_top35_feature_list.csv
./cd_catboost_feature_selection/QY_%_best_top35_feature_list.csv
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from model.model import data_process, CarbonDotFeaturizer


TARGETS = ["Ex_max_nm", "Em_max_nm", "QY_%"]
ROW_ID_COL = "initial_row_id"
N_BITS = 256
RADIUS = 2
DEFAULT_REAGENT_1_ML = 15.0

FEATURE_LIST_FILES = {
    "Ex_max_nm": "./cd_catboost_feature_selection/Ex_max_nm_best_top20_feature_list.csv",
    "Em_max_nm": "./cd_catboost_feature_selection/Em_max_nm_best_top35_feature_list.csv",
    "QY_%": "./cd_catboost_feature_selection/QY_%_best_top35_feature_list.csv",
}

FORBIDDEN_FEATURES = {
    "Ex_max_nm", "Em_max_nm", "QY_%", "stoke_shift",
    "Stokes_shift_nm", "stoke_shift_calc_check", "success",
}


@dataclass
class Cfg:
    initial_data_file: str = "./data/output_with_molar_amount_pubchem.xlsx"
    smiles_file: str = "./data/compound_smiles.xlsx"
    output_dir: str = "./success1_generated_feature_model_benchmark"

    n_splits: int = 5
    n_repeats: int = 3
    cv_seed: int = 42

    nn_epochs: int = 300
    nn_batch_size: int = 32
    nn_lr: float = 2e-3
    nn_weight_decay: float = 1e-4
    nn_patience: int = 30
    nn_grad_clip: float = 1.0

    cb_iterations: int = 2000
    cb_learning_rate: float = 0.05
    cb_depth: int = 4
    cb_l2_leaf_reg: float = 5.0
    cb_random_strength: float = 1.0
    cb_subsample: float = 0.9
    cb_early_stopping_rounds: int = 100

    dpi: int = 300
    # Compare all schemes with identical models/feature lists and training budgets.
    # "fixed" means NO nested hyperparameter or feature selection is claimed.
    split_methods: tuple = ("random", "pair_grouped", "leave_one_pair_out")
    grouped_repeats: int = 3
    run_leave_one_pair_out: bool = True  # 121 pairs means 121 additional FPNN fits; set False to skip


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Embedded CatBoost feature engineering (copied from the revised 02 workflow)
# ---------------------------------------------------------------------------
def norm_name(value) -> str:
    return str(value).strip().lower()


def safe_div(a, b, eps=1e-8):
    return a / (b + eps)


def ensure_shift_and_success_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "stoke_shift" not in out.columns:
        if "Stokes_shift_nm" in out.columns:
            out["stoke_shift"] = out["Stokes_shift_nm"]
        elif {"Ex_max_nm", "Em_max_nm"}.issubset(out.columns):
            out["stoke_shift"] = out["Em_max_nm"] - out["Ex_max_nm"]
    if "Stokes_shift_nm" not in out.columns and "stoke_shift" in out.columns:
        out["Stokes_shift_nm"] = out["stoke_shift"]
    if "success" in out.columns:
        out["success"] = pd.to_numeric(out["success"], errors="coerce")
    return out


def mol_from_smiles(smiles):
    return None if pd.isna(smiles) else Chem.MolFromSmiles(str(smiles))


def element_count(mol, atomic_num):
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == atomic_num)


def substructure_count(mol, smarts):
    patt = Chem.MolFromSmarts(smarts)
    return 0 if patt is None else len(mol.GetSubstructMatches(patt))


def has_substructure(mol, smarts):
    patt = Chem.MolFromSmarts(smarts)
    return 0 if patt is None else int(mol.HasSubstructMatch(patt))


FG_SMARTS = {
    "fg_hydroxyl": "[OX2H]", "fg_phenol": "c[OX2H]",
    "fg_carboxylic_acid": "[CX3](=O)[OX2H1]", "fg_ester": "[CX3](=O)[OX2][#6]",
    "fg_aldehyde": "[CX3H1](=O)[#6]", "fg_ketone": "[#6][CX3](=O)[#6]",
    "fg_carbonyl_any": "[CX3]=[OX1]", "fg_primary_amine": "[NX3;H2][#6]",
    "fg_secondary_amine": "[NX3;H1]([#6])[#6]",
    "fg_tertiary_amine": "[NX3;H0]([#6])([#6])[#6]",
    "fg_amide": "[NX3][CX3](=[OX1])[#6]",
    "fg_imide_or_urea_like": "[NX3][CX3](=[OX1])[NX3]",
    "fg_nitro": "[$([NX3](=O)=O),$([N+](=O)[O-])]", "fg_nitrile": "[CX2]#N",
    "fg_thiol": "[SX2H]", "fg_sulfide": "[#16X2][#6]",
    "fg_sulfonic_acid": "[SX4](=O)(=O)[OX2H]", "fg_phosphate_like": "P(=O)(O)(O)",
    "fg_aromatic_ring": "a1aaaaa1", "fg_heteroaromatic": "[a;r][n,o,s]",
}


def calc_mol_features(mol, prefix=""):
    if mol is None:
        raise ValueError("RDKit failed to parse a precursor SMILES")
    f = {
        f"{prefix}MolWt": Descriptors.MolWt(mol),
        f"{prefix}ExactMolWt": Descriptors.ExactMolWt(mol),
        f"{prefix}HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        f"{prefix}NumValenceElectrons": Descriptors.NumValenceElectrons(mol),
        f"{prefix}NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        f"{prefix}NumAliphaticRings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        f"{prefix}NumSaturatedRings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        f"{prefix}NumHAcceptors": Lipinski.NumHAcceptors(mol),
        f"{prefix}NumHDonors": Lipinski.NumHDonors(mol),
        f"{prefix}NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
        f"{prefix}TPSA": rdMolDescriptors.CalcTPSA(mol),
        f"{prefix}MolLogP": Crippen.MolLogP(mol),
        f"{prefix}FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        f"{prefix}FormalCharge": Chem.GetFormalCharge(mol),
        f"{prefix}RingCount": rdMolDescriptors.CalcNumRings(mol),
        f"{prefix}HeteroAtomCount": rdMolDescriptors.CalcNumHeteroatoms(mol),
        f"{prefix}NumAtoms": mol.GetNumAtoms(),
    }
    counts = {"C": 6, "N": 7, "O": 8, "S": 16, "P": 15}
    vals = {k: element_count(mol, z) for k, z in counts.items()}
    vals["H"] = sum(a.GetTotalNumHs(includeNeighbors=True) for a in mol.GetAtoms())
    for key, value in vals.items():
        f[f"{prefix}{key}_count"] = value
    for key in ["N", "O", "S", "P"]:
        f[f"{prefix}{key}_to_C"] = safe_div(vals[key], vals["C"])
    f[f"{prefix}Hetero_to_C"] = safe_div(
        vals["N"] + vals["O"] + vals["S"] + vals["P"], vals["C"]
    )
    for name, smarts in FG_SMARTS.items():
        f[f"{prefix}{name}_count"] = substructure_count(mol, smarts)
        f[f"{prefix}{name}_has"] = has_substructure(mol, smarts)
    return f


def calc_fp_bits(mol, prefix="", n_bits=N_BITS, radius=RADIUS):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return fp, {f"{prefix}fp_{i}": int(v) for i, v in enumerate(arr)}


def solvent_feature_dict():
    base = {
        "water": (1, 0, 0, 0, 0, 1, 0, 78.5, 100.0, 9.0, 0),
        "ethanol": (0, 1, 0, 0, 0, 1, 0, 24.3, 78.4, 5.2, 0),
        "dimethylformamide": (0, 0, 1, 0, 0, 0, 1, 36.7, 153.0, 6.4, 0),
        "phosphoric acid": (0, 0, 0, 1, 1, 1, 0, 61.0, 158.0, 8.0, 2),
        "sulfuric acid": (0, 0, 0, 1, 1, 1, 0, 100.0, 337.0, 9.5, 3),
    }
    names = ["solv_is_water", "solv_is_alcohol", "solv_is_amide_solvent",
             "solv_is_acid", "solv_is_strong_acid", "solv_is_protic",
             "solv_has_n", "solv_dielectric", "solv_bp_C",
             "solv_polarity_index", "solv_relative_acidity_level"]
    return {k: dict(zip(names, v)) for k, v in base.items()}


def get_solvent_polarity_data():
    return {
        "water": {"dielectric_constant": 80.1, "pi_star": 1.09, "HBD": 1.17, "HBA": 0.47, "ET30": 63.1, "pKa": 15.7, "ORP_mV": 300, "density_g_cm3": 0.997, "specific_heat_J_gK": 4.18, "viscosity_mPa_s": 0.89, "boiling_point_C": 100},
        "dimethylformamide": {"dielectric_constant": 36.7, "pi_star": 0.88, "HBD": 0.00, "HBA": 0.69, "ET30": 43.2, "pKa": 29, "ORP_mV": 200, "density_g_cm3": 0.944, "specific_heat_J_gK": 2.47, "viscosity_mPa_s": 0.92, "boiling_point_C": 153},
        "ethanol": {"dielectric_constant": 24.6, "pi_star": 0.54, "HBD": 0.86, "HBA": 0.75, "ET30": 51.9, "pKa": 15.9, "ORP_mV": 250, "density_g_cm3": 0.789, "specific_heat_J_gK": 2.44, "viscosity_mPa_s": 1.07, "boiling_point_C": 78.37},
        "sulfuric acid": {"dielectric_constant": 83, "pi_star": 1.11, "HBD": 1.22, "HBA": 0.48, "ET30": 64.8, "pKa": 0.0, "ORP_mV": 800, "density_g_cm3": 1.05, "specific_heat_J_gK": 3.6, "viscosity_mPa_s": 1.2, "boiling_point_C": 100},
        "phosphoric acid": {"dielectric_constant": 82, "pi_star": 1.10, "HBD": 1.20, "HBA": 0.47, "ET30": 64.0, "pKa": 1.0, "ORP_mV": 700, "density_g_cm3": 1.03, "specific_heat_J_gK": 3.8, "viscosity_mPa_s": 1.1, "boiling_point_C": 100},
    }


def build_compound_feature_table(df_smiles, n_bits=N_BITS):
    rows, fp_objects = [], {}
    for _, row in df_smiles.iterrows():
        name, smiles = row["compound_name"], row["SMILES"]
        mol = mol_from_smiles(smiles)
        if mol is None:
            warnings.warn(f"Cannot parse SMILES for {name}: {smiles}")
            continue
        features = {"compound_name": name, "compound_key": norm_name(name), "SMILES": smiles}
        features.update(calc_mol_features(mol))
        fp, bits = calc_fp_bits(mol, n_bits=n_bits)
        features.update(bits)
        rows.append(features)
        fp_objects[norm_name(name)] = fp
    return pd.DataFrame(rows), fp_objects


def standardize_candidate_columns(df):
    out = df.copy()
    aliases = {"reagent": "reagent_1", "mass_1": "mass_1_g", "mass_2": "mass_2_g"}
    for old, new in aliases.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})
    if "Mass1(g)" not in out.columns:
        if "mass_1_g" in out.columns: out["Mass1(g)"] = out["mass_1_g"]
        elif "mass_1_mg" in out.columns: out["Mass1(g)"] = out["mass_1_mg"] / 1000
    if "Mass2(g)" not in out.columns:
        if "mass_2_g" in out.columns: out["Mass2(g)"] = out["mass_2_g"]
        elif "mass_2_mg" in out.columns: out["Mass2(g)"] = out["mass_2_mg"] / 1000
    if "reagent_1_mL" not in out.columns:
        out["reagent_1_mL"] = DEFAULT_REAGENT_1_ML
    required = ["precursor_1", "precursor_2", "reagent_1", "pre_1_mmol",
                "pre_2_mmol", "Mass1(g)", "Mass2(g)", "reagent_1_mL",
                "temperature", "time_h"]
    missing = [c for c in required if c not in out.columns]
    if missing: raise ValueError(f"Input data missing feature-engineering columns: {missing}")
    return out


def add_qy_extra_features(out):
    def col(name, default=0.0):
        return out[name] if name in out.columns else default
    hetero = col("pair_total_N") + col("pair_total_O") + col("pair_total_S") + col("pair_total_P")
    formulas = {
        "qy_pair_total_hetero": hetero,
        "qy_pair_hetero_to_C": hetero / (col("pair_total_C") + 1e-8),
        "qy_acidity_x_temp": col("solv_relative_acidity_level") * col("temperature"),
        "qy_acidity_x_time": col("solv_relative_acidity_level") * col("time_h"),
        "qy_acidity_x_temp_time": col("solv_relative_acidity_level") * col("temp_time"),
        "qy_protic_x_temp_time": col("solv_is_protic") * col("temp_time"),
        "qy_acid_x_pair_total_N": col("solv_relative_acidity_level") * col("pair_total_N"),
        "qy_acid_x_pair_total_O": col("solv_relative_acidity_level") * col("pair_total_O"),
        "qy_acid_x_pair_total_S": col("solv_relative_acidity_level") * col("pair_total_S"),
        "qy_acid_x_pair_total_P": col("solv_relative_acidity_level") * col("pair_total_P"),
        "qy_temp_x_pair_total_hetero": col("temperature") * hetero,
        "qy_time_x_pair_total_hetero": col("time_h") * hetero,
        "qy_temp_time_x_pair_total_hetero": col("temp_time") * hetero,
        "qy_acid_x_pair_N_to_C": col("solv_relative_acidity_level") * col("pair_N_to_C"),
        "qy_acid_x_pair_O_to_C": col("solv_relative_acidity_level") * col("pair_O_to_C"),
        "qy_acid_x_pair_S_to_C": col("solv_relative_acidity_level") * col("pair_S_to_C"),
        "qy_acid_x_pair_P_to_C": col("solv_relative_acidity_level") * col("pair_P_to_C"),
        "qy_acid_x_amine_plus_acid": col("solv_relative_acidity_level") * col("pair_amine_plus_acid"),
        "qy_protic_x_amine_plus_acid": col("solv_is_protic") * col("pair_amine_plus_acid"),
        "qy_temp_x_tanimoto": col("temperature") * col("fp_tanimoto"),
        "qy_time_x_tanimoto": col("time_h") * col("fp_tanimoto"),
        "qy_acid_x_tanimoto": col("solv_relative_acidity_level") * col("fp_tanimoto"),
        "qy_reagent_ratio_x_temp": col("reagent_to_precursor_vol_mass_ratio") * col("temperature"),
        "qy_reagent_ratio_x_time": col("reagent_to_precursor_vol_mass_ratio") * col("time_h"),
        "qy_reagent_ratio_x_hetero": col("reagent_to_precursor_vol_mass_ratio") * hetero,
        "qy_mass_ratio_x_acidity": col("mass_ratio_12") * col("solv_relative_acidity_level"),
        "qy_mmol_ratio_x_acidity": col("pre_mmol_ratio_12") * col("solv_relative_acidity_level"),
    }
    for name, value in formulas.items():
        out[name] = np.asarray(value, dtype=np.float32)
    return out


def build_dataset_features(exp_df, compound_feat_df, fp_objs, has_targets=True):
    exp = standardize_candidate_columns(exp_df)
    exp["precursor_1_key"] = exp["precursor_1"].map(norm_name)
    exp["precursor_2_key"] = exp["precursor_2"].map(norm_name)
    exp["reagent_1_key"] = exp["reagent_1"].map(norm_name)
    out = exp.merge(compound_feat_df.add_prefix("p1_"), left_on="precursor_1_key",
                    right_on="p1_compound_key", how="left")
    out = out.merge(compound_feat_df.add_prefix("p2_"), left_on="precursor_2_key",
                    right_on="p2_compound_key", how="left")
    if out[["p1_SMILES", "p2_SMILES"]].isna().any().any():
        raise ValueError("At least one precursor could not be mapped to compound_smiles.xlsx")

    solvent_map = solvent_feature_dict()
    missing_solvents = sorted(set(out["reagent_1_key"]) - set(solvent_map))
    if missing_solvents:
        raise KeyError(f"Solvents absent from embedded solvent table: {missing_solvents}")
    out = pd.concat([
        out,
        pd.DataFrame([solvent_map[x] for x in out["reagent_1_key"]], index=out.index)
    ], axis=1)

    out["pre_mmol_sum"] = out["pre_1_mmol"] + out["pre_2_mmol"]
    out["pre_mmol_diff_abs"] = (out["pre_1_mmol"] - out["pre_2_mmol"]).abs()
    out["pre_mmol_ratio_12"] = safe_div(out["pre_1_mmol"], out["pre_2_mmol"])
    out["pre_mmol_ratio_21"] = safe_div(out["pre_2_mmol"], out["pre_1_mmol"])
    out["pre1_mmol_frac"] = safe_div(out["pre_1_mmol"], out["pre_mmol_sum"])
    out["pre2_mmol_frac"] = safe_div(out["pre_2_mmol"], out["pre_mmol_sum"])
    out["mass_sum_g"] = out["Mass1(g)"] + out["Mass2(g)"]
    out["mass_diff_abs_g"] = (out["Mass1(g)"] - out["Mass2(g)"]).abs()
    out["mass_ratio_12"] = safe_div(out["Mass1(g)"], out["Mass2(g)"])
    out["mass_ratio_21"] = safe_div(out["Mass2(g)"], out["Mass1(g)"])
    out["reagent_to_precursor_mass_ratio"] = safe_div(out["reagent_1_mL"], out["mass_sum_g"])
    out["reagent_to_precursor_vol_mass_ratio"] = out["reagent_to_precursor_mass_ratio"]
    out["temp_time"] = out["temperature"] * out["time_h"]
    out["temp_over_time"] = safe_div(out["temperature"], out["time_h"])
    out["same_precursor_flag"] = (out["precursor_1_key"] == out["precursor_2_key"]).astype(int)

    descriptor_names = [
        "MolWt", "ExactMolWt", "HeavyAtomCount", "NumValenceElectrons",
        "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
        "NumHAcceptors", "NumHDonors", "NumRotatableBonds", "TPSA", "MolLogP",
        "FractionCSP3", "FormalCharge", "RingCount", "HeteroAtomCount", "NumAtoms",
        "C_count", "H_count", "N_count", "O_count", "S_count", "P_count",
        "N_to_C", "O_to_C", "S_to_C", "P_to_C", "Hetero_to_C",
    ]
    descriptor_names += [f"{name}_{suffix}" for name in FG_SMARTS for suffix in ("count", "has")]
    for name in descriptor_names:
        a, b = out[f"p1_{name}"], out[f"p2_{name}"]
        out[f"pair_sum_{name}"] = a + b
        out[f"pair_absdiff_{name}"] = (a - b).abs()
        out[f"pair_prod_{name}"] = a * b
    for name in ["MolWt", "N_count", "O_count", "S_count", "P_count", "TPSA", "MolLogP"]:
        out[f"pair_ratio12_{name}"] = safe_div(out[f"p1_{name}"], out[f"p2_{name}"])
        out[f"pair_ratio21_{name}"] = safe_div(out[f"p2_{name}"], out[f"p1_{name}"])

    for element in ["N", "O", "S", "P", "C"]:
        out[f"pair_total_{element}"] = out[f"p1_{element}_count"] + out[f"p2_{element}_count"]
    for element in ["N", "O", "S", "P"]:
        out[f"pair_{element}_to_C"] = safe_div(out[f"pair_total_{element}"], out["pair_total_C"])
    out["pair_has_N_and_O"] = ((out["pair_total_N"] > 0) & (out["pair_total_O"] > 0)).astype(int)
    out["pair_has_N_and_S"] = ((out["pair_total_N"] > 0) & (out["pair_total_S"] > 0)).astype(int)
    out["pair_has_N_and_P"] = ((out["pair_total_N"] > 0) & (out["pair_total_P"] > 0)).astype(int)
    out["pair_both_aromatic"] = ((out["p1_fg_aromatic_ring_has"] > 0) & (out["p2_fg_aromatic_ring_has"] > 0)).astype(int)
    out["pair_one_aromatic_one_nonaromatic"] = ((out["p1_fg_aromatic_ring_has"] > 0) ^ (out["p2_fg_aromatic_ring_has"] > 0)).astype(int)
    amine1 = out[["p1_fg_primary_amine_has", "p1_fg_secondary_amine_has", "p1_fg_tertiary_amine_has"]].sum(axis=1) > 0
    amine2 = out[["p2_fg_primary_amine_has", "p2_fg_secondary_amine_has", "p2_fg_tertiary_amine_has"]].sum(axis=1) > 0
    out["pair_amine_plus_acid"] = ((amine1 & (out["p2_fg_carboxylic_acid_has"] > 0)).astype(int) +
                                    (amine2 & (out["p1_fg_carboxylic_acid_has"] > 0)).astype(int))

    tanimoto, intersection, union, xor = [], [], [], []
    for _, row in out.iterrows():
        fp1, fp2 = fp_objs[row["precursor_1_key"]], fp_objs[row["precursor_2_key"]]
        tanimoto.append(DataStructs.TanimotoSimilarity(fp1, fp2))
        a = np.fromiter(fp1.ToBitString(), dtype=np.int8)
        b = np.fromiter(fp2.ToBitString(), dtype=np.int8)
        intersection.append(np.logical_and(a, b).sum())
        union.append(np.logical_or(a, b).sum())
        xor.append(np.logical_xor(a, b).sum())
    out["fp_tanimoto"] = tanimoto
    out["fp_intersection_count"] = intersection
    out["fp_union_count"] = union
    out["fp_xor_count"] = xor
    out = add_qy_extra_features(out)
    if has_targets:
        out["stoke_shift"] = out["Em_max_nm"] - out["Ex_max_nm"]
    return out


def read_feature_list(path: str) -> list[str]:
    table = read_table(path)
    if "feature" not in table.columns:
        raise ValueError(f"Feature-list file must contain column 'feature': {path}")
    cols = list(dict.fromkeys(table["feature"].dropna().astype(str).str.strip()))
    cols = [c for c in cols if c]
    leakage = [c for c in cols if c in FORBIDDEN_FEATURES]
    if leakage:
        raise ValueError(f"Target leakage in {path}: {leakage}")
    return cols


def load_success1_data(cfg: Cfg) -> pd.DataFrame:
    """Add stable row IDs, filter success=1, then apply property preprocessing."""
    raw = read_table(cfg.initial_data_file).copy()
    if "success" not in raw.columns:
        raise ValueError("Initial dataset must contain a 'success' column.")

    raw[ROW_ID_COL] = np.arange(len(raw), dtype=int)
    raw["success"] = pd.to_numeric(raw["success"], errors="coerce")
    counts = raw["success"].value_counts(dropna=False).to_dict()
    filtered = raw.loc[raw["success"].eq(1)].copy()
    if filtered.empty:
        raise ValueError("No rows with success == 1 were found.")

    filtered = ensure_shift_and_success_columns(filtered)
    processed = data_process(filtered).copy()

    # Some data_process implementations reset or drop metadata. Restore the
    # stable ID only when row order and length are demonstrably unchanged.
    if ROW_ID_COL not in processed.columns:
        if len(processed) != len(filtered):
            raise ValueError(
                "data_process removed rows and did not preserve initial_row_id; "
                "safe alignment cannot be established. Modify data_process to "
                "retain initial_row_id."
            )
        processed[ROW_ID_COL] = filtered[ROW_ID_COL].to_numpy()

    processed = ensure_shift_and_success_columns(processed)
    required = [
        ROW_ID_COL, "precursor_1", "precursor_2", "reagent_1",
        "temperature", "time_h", "pre_1_mmol", "pre_2_mmol", *TARGETS,
    ]
    missing = [c for c in required if c not in processed.columns]
    if missing:
        raise ValueError(f"Processed success=1 data missing columns: {missing}")

    processed = processed.dropna(subset=required).reset_index(drop=True)
    processed[ROW_ID_COL] = processed[ROW_ID_COL].astype(int)
    if processed[ROW_ID_COL].duplicated().any():
        raise ValueError("Duplicated initial_row_id after preprocessing.")

    print(f"Initial rows: {len(raw)}; success counts: {counts}")
    print(f"Usable success=1 property rows: {len(processed)}")
    return processed


def generate_selected_features(
    data: pd.DataFrame, cfg: Cfg
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, pd.DataFrame]]:
    """Generate full features in memory, then retain target-specific columns."""
    smiles_compound = pd.read_excel(cfg.smiles_file, sheet_name="compound")
    compound_features, fp_objects = build_compound_feature_table(smiles_compound)
    full = build_dataset_features(
        data.copy(), compound_features, fp_objects, has_targets=True
    )

    if len(full) != len(data):
        raise ValueError("Feature generation changed the number of rows.")
    if ROW_ID_COL not in full.columns:
        full[ROW_ID_COL] = data[ROW_ID_COL].to_numpy()
    if not np.array_equal(full[ROW_ID_COL].astype(int), data[ROW_ID_COL].astype(int)):
        raise ValueError("Row order changed during feature generation.")

    feature_lists: dict[str, list[str]] = {}
    selected: dict[str, pd.DataFrame] = {}
    for target in TARGETS:
        cols = read_feature_list(FEATURE_LIST_FILES[target])
        missing = [c for c in cols if c not in full.columns]
        if missing:
            raise ValueError(
                f"Generated feature table lacks {len(missing)} selected features "
                f"for {target}. Examples: {missing[:20]}"
            )
        X = full.loc[:, cols].copy()
        for col in cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        feature_lists[target] = cols
        selected[target] = X
        print(f"{target}: generated and selected {len(cols)} features")
    return full, feature_lists, selected


class FPProcessDataset(Dataset):
    def __init__(self, df: pd.DataFrame, featurizer):
        self.df = df.reset_index(drop=True)
        self.featurizer = featurizer
        precursors = sorted(
            set(self.df["precursor_1"].astype(str)) |
            set(self.df["precursor_2"].astype(str))
        )
        self.fp_map = {
            p: torch.tensor(featurizer.fingerprints(p), dtype=torch.float32)
            for p in precursors
        }
        reagents = sorted(set(self.df["reagent_1"].astype(str)))
        self.solvent_map = {r: featurizer.solvent_feat(r).float() for r in reagents}

        # Production 02 model uses fixed input dimensions. Fail early if the
        # currently installed model.model featurizer is incompatible.
        if self.fp_map:
            fp_dim = int(next(iter(self.fp_map.values())).numel())
            if fp_dim != 679:
                raise ValueError(
                    f"02 FPNN requires a 679-dimensional fingerprint, got {fp_dim}."
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        pre1 = float(r["pre_1_mmol"])
        pre2 = float(r["pre_2_mmol"])
        ratio = pre1 / (pre1 + pre2) if pre1 + pre2 > 0 else 0.5
        global_feat = torch.cat([
            self.solvent_map[str(r["reagent_1"])],
            self.featurizer.reaction_feat(
                float(r["temperature"]), float(r["time_h"])
            ).float(),
            torch.tensor([ratio], dtype=torch.float32),
        ])
        if global_feat.numel() != 14:
            raise ValueError(
                "02 FPNN requires global_feat dimension 14 "
                f"(solvent + reaction + ratio), got {global_feat.numel()}."
            )
        return {
            "fp1": self.fp_map[str(r["precursor_1"])],
            "fp2": self.fp_map[str(r["precursor_2"])],
            "global": global_feat,
            "y": torch.tensor(r[TARGETS].astype(float).to_numpy(), dtype=torch.float32),
            "row_id": int(r[ROW_ID_COL]),
        }


def collate(batch):
    return {
        "fp1": torch.stack([x["fp1"] for x in batch]),
        "fp2": torch.stack([x["fp2"] for x in batch]),
        "global": torch.stack([x["global"] for x in batch]),
        "y": torch.stack([x["y"] for x in batch]),
        "row_id": np.asarray([x["row_id"] for x in batch], dtype=int),
    }


class FingerprintEncoderSmall(nn.Module):
    """Exact fingerprint encoder used in 02_train_fp_catboost_cycle.py."""
    def __init__(self, input_dim=679, latent_dim=32, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, latent_dim), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x.float())


class ProcessEncoderSmall(nn.Module):
    """Exact 14-to-16 process encoder used in the revised 02 model."""
    def __init__(self, input_dim=14, out_dim=16, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x.float())


class FPProcessNN(nn.Module):
    """Exact multi-output FPNN architecture from the production 02 script."""
    def __init__(self):
        super().__init__()
        self.fp_encoder = FingerprintEncoderSmall(
            input_dim=679, latent_dim=32, dropout=0.30
        )
        self.proc_encoder = ProcessEncoderSmall(
            input_dim=14, out_dim=16, dropout=0.10
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(64, 64), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(80, 64), nn.GELU(), nn.Dropout(0.30), nn.Linear(64, 3)
        )

    def forward(self, fps_1, fps_2, global_feat):
        f1 = self.fp_encoder(fps_1)
        f2 = self.fp_encoder(fps_2)
        fusion = self.fusion_mlp(
            torch.cat([f1 + f2, torch.abs(f1 - f2)], dim=-1)
        )
        proc = self.proc_encoder(global_feat)
        return self.head(torch.cat([fusion, proc], dim=-1))


def predict_nn(model, loader, scaler, device) -> np.ndarray:
    model.eval()
    rows = []
    with torch.no_grad():
        for b in loader:
            p = model(
                b["fp1"].to(device), b["fp2"].to(device), b["global"].to(device)
            ).cpu().numpy()
            rows.append(p)
    return scaler.inverse_transform(np.vstack(rows)).astype(np.float32)


def fit_predict_nn(dataset, tr, te, cfg: Cfg, seed: int):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = StandardScaler().fit(dataset.df.loc[tr, TARGETS].astype(float))
    train_loader = DataLoader(
        Subset(dataset, tr), batch_size=cfg.nn_batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
    )
    train_eval_loader = DataLoader(
        Subset(dataset, tr), batch_size=cfg.nn_batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )
    test_loader = DataLoader(
        Subset(dataset, te), batch_size=cfg.nn_batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )
    model = FPProcessNN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.nn_lr, weight_decay=cfg.nn_weight_decay
    )

    # Use a deterministic 15% validation subset drawn only from the outer train
    # fold. The outer test fold never controls early stopping.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(np.asarray(tr, dtype=int))
    n_val = max(1, int(round(0.15 * len(perm))))
    val_idx = perm[:n_val].tolist()
    fit_idx = perm[n_val:].tolist()
    fit_loader = DataLoader(
        Subset(dataset, fit_idx), batch_size=cfg.nn_batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=cfg.nn_batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    best_loss, bad, best_state = np.inf, 0, None
    for _ in range(cfg.nn_epochs):
        model.train()
        for b in fit_loader:
            ys = scaler.transform(b["y"].numpy()).astype(np.float32)
            pred = model(
                b["fp1"].to(device), b["fp2"].to(device), b["global"].to(device)
            )
            loss = F.mse_loss(pred, torch.tensor(ys, device=device))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.nn_grad_clip)
            optimizer.step()

        model.eval()
        losses = []
        with torch.no_grad():
            for b in val_loader:
                ys = scaler.transform(b["y"].numpy()).astype(np.float32)
                pred = model(
                    b["fp1"].to(device), b["fp2"].to(device), b["global"].to(device)
                )
                losses.append(F.mse_loss(pred, torch.tensor(ys, device=device)).item())
        value = float(np.mean(losses))
        if value < best_loss - 1e-7:
            best_loss = value
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.nn_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return (
        predict_nn(model, train_eval_loader, scaler, device),
        predict_nn(model, test_loader, scaler, device),
    )


def make_catboost_model(seed: int, cfg: Cfg):
    """Exact CatBoost regression settings used by the revised 02 workflow."""
    return CatBoostRegressor(
        loss_function="RMSE", eval_metric="RMSE",
        iterations=cfg.cb_iterations, learning_rate=cfg.cb_learning_rate,
        depth=cfg.cb_depth, l2_leaf_reg=cfg.cb_l2_leaf_reg,
        random_strength=cfg.cb_random_strength, bootstrap_type="Bernoulli",
        subsample=cfg.cb_subsample, grow_policy="SymmetricTree",
        random_seed=seed, verbose=False, allow_writing_files=False,
    )


def fit_predict_catboost(
    selected_features, y, tr, te, cfg, seed
) -> tuple[np.ndarray, np.ndarray]:
    train_pred = np.zeros((len(tr), len(TARGETS)), dtype=np.float32)
    test_pred = np.zeros((len(te), len(TARGETS)), dtype=np.float32)
    for j, target in enumerate(TARGETS):
        X = selected_features[target]
        model = make_catboost_model(seed + 1000 * j, cfg)
        # Use an inner validation subset so the outer test fold remains fully
        # independent. This changes only CV control, not the 02 model family.
        rng = np.random.default_rng(seed + j)
        shuffled = rng.permutation(np.asarray(tr, dtype=int))
        n_val = max(1, int(round(0.15 * len(shuffled))))
        inner_val = shuffled[:n_val]
        inner_fit = shuffled[n_val:]
        model.fit(
            X.iloc[inner_fit], y[inner_fit, j],
            eval_set=(X.iloc[inner_val], y[inner_val, j]),
            use_best_model=True,
            early_stopping_rounds=cfg.cb_early_stopping_rounds,
            verbose=False,
        )
        train_pred[:, j] = model.predict(X.iloc[tr])
        test_pred[:, j] = model.predict(X.iloc[te])
    return train_pred, test_pred


def metrics(y_true, pred) -> dict[str, float]:
    out = {}
    r2_values = []
    for j, target in enumerate(TARGETS):
        # In LOPPO, most pairs have only one test sample. Per-fold R2 is undefined.
        out[f"{target}_R2"] = (r2_score(y_true[:, j], pred[:, j])
                               if len(y_true) >= 2 else np.nan)
        out[f"{target}_MAE"] = mean_absolute_error(y_true[:, j], pred[:, j])
        out[f"{target}_RMSE"] = np.sqrt(mean_squared_error(y_true[:, j], pred[:, j]))
        r2_values.append(out[f"{target}_R2"])
    out["macro_R2"] = (float(np.nanmean(r2_values))
                       if np.isfinite(r2_values).any() else np.nan)
    return out


def canonical_pair_groups(data: pd.DataFrame) -> np.ndarray:
    """Order-invariant precursor pairs; normalized spelling and whitespace."""
    a = data["precursor_1"].map(norm_name)
    b = data["precursor_2"].map(norm_name)
    if a.eq("nan").any() or b.eq("nan").any() or a.eq("").any() or b.eq("").any():
        raise ValueError("Invalid or missing precursor identity; cannot make groups.")
    return np.array([" || ".join(sorted((x, y))) for x, y in zip(a, b)], dtype=str)


def make_splits(data: pd.DataFrame, groups: np.ndarray, cfg: Cfg):
    """Return (split_method, repeat, fold, train_idx, test_idx).

    Grouped CV: shuffled whole precursor pairs, balanced by their record counts.
    Groups are never split across train and test. No label y is consulted.
    """
    if "random" in cfg.split_methods:
        cv = RepeatedKFold(n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.cv_seed)
        for i, (tr, te) in enumerate(cv.split(data)):
            yield "random", i // cfg.n_splits + 1, i % cfg.n_splits + 1, tr, te

    unique, counts = np.unique(groups, return_counts=True)
    if len(unique) < cfg.n_splits and "pair_grouped" in cfg.split_methods:
        raise ValueError(f"Need >= {cfg.n_splits} unique precursor pairs; found {len(unique)}.")
    if "pair_grouped" in cfg.split_methods:
        for repeat in range(1, cfg.grouped_repeats + 1):
            rng = np.random.default_rng(cfg.cv_seed + repeat * 1009)
            # Shuffle first for random tie-breaking, then place largest groups first.
            order = rng.permutation(len(unique))
            order = order[np.argsort(-counts[order], kind="stable")]
            folds, sizes = [[] for _ in range(cfg.n_splits)], np.zeros(cfg.n_splits, dtype=int)
            for i in order:
                j = int(np.argmin(sizes))
                folds[j].append(unique[i]); sizes[j] += int(counts[i])
            for fold, held_out in enumerate(folds, start=1):
                te = np.flatnonzero(np.isin(groups, held_out))
                tr = np.flatnonzero(~np.isin(groups, held_out))
                if len(tr) == 0 or len(te) == 0: raise ValueError("Empty grouped split")
                yield "pair_grouped", repeat, fold, tr, te

    if "leave_one_pair_out" in cfg.split_methods and cfg.run_leave_one_pair_out:
        for fold, group in enumerate(unique, start=1):
            te = np.flatnonzero(groups == group)
            tr = np.flatnonzero(groups != group)
            if len(tr) == 0: raise ValueError("Leave-one-pair-out requires >=2 groups")
            yield "leave_one_pair_out", 1, fold, tr, te


def run_cv(data, dataset, selected_features, cfg: Cfg, output: Path):
    y = data[TARGETS].astype(float).to_numpy(dtype=np.float32)
    groups = canonical_pair_groups(data)
    pd.DataFrame({ROW_ID_COL: data[ROW_ID_COL], "precursor_pair_group": groups,
                  "precursor_1": data["precursor_1"], "precursor_2": data["precursor_2"]}).to_csv(
        output / "precursor_pair_groups.csv", index=False, encoding="utf-8-sig")
    fold_rows, prediction_rows, audit_rows = [], [], []
    model_names = ["FPNN", "CatBoost", "0.3FPNN-0.7CatBoost", "0.5FPNN-0.5CatBoost", "0.7FPNN-0.3CatBoost"]

    for run, (method, repeat, fold, tr, te) in enumerate(make_splits(data, groups, cfg), start=1):
        overlap = set(groups[tr]) & set(groups[te])
        if method != "random" and overlap: raise AssertionError(f"Group leakage: {overlap}")
        seed = cfg.cv_seed + run * 100
        print(f"[{method}, repeat={repeat}, fold={fold}] n_train={len(tr)} n_test={len(te)} "
              f"train_pairs={len(set(groups[tr]))} test_pairs={len(set(groups[te]))} pair_overlap={len(overlap)}")
        audit_rows.append({"method": method, "repeat": repeat, "fold": fold, "run": run,
                           "n_train": len(tr), "n_test": len(te), "n_train_pairs": len(set(groups[tr])),
                           "n_test_pairs": len(set(groups[te])), "n_overlapping_pairs": len(overlap),
                           "train_row_ids": ";".join(map(str, data.iloc[tr][ROW_ID_COL])),
                           "test_row_ids": ";".join(map(str, data.iloc[te][ROW_ID_COL])),
                           "test_pairs": ";".join(sorted(set(groups[te])))})
        nn_tr, nn_te = fit_predict_nn(dataset, tr.tolist(), te.tolist(), cfg, seed)
        cb_tr, cb_te = fit_predict_catboost(selected_features, y, tr, te, cfg, seed)
        predictions = {"FPNN": (nn_tr, nn_te), "CatBoost": (cb_tr, cb_te)}
        for w in (0.3, 0.5, 0.7):
            predictions[f"{w:.1f}FPNN-{1-w:.1f}CatBoost"] = (w * nn_tr + (1-w) * cb_tr,
                                                               w * nn_te + (1-w) * cb_te)
        for model_name, (pred_tr, pred_te) in predictions.items():
            for split, indices, pred in (("train", tr, pred_tr), ("test", te, pred_te)):
                row = {"method": method, "repeat": repeat, "fold": fold, "run": run,
                       "split": split, "model": model_name, "n_train": len(tr), "n_test": len(te)}
                row.update(metrics(y[indices], pred)); fold_rows.append(row)
            for local, pos in enumerate(te):
                rec = {"method": method, "repeat": repeat, "fold": fold, "run": run,
                       "model": model_name, ROW_ID_COL: int(data.iloc[pos][ROW_ID_COL]),
                       "precursor_pair_group": groups[pos]}
                for j, target in enumerate(TARGETS):
                    rec[f"true_{target}"] = float(y[pos, j]); rec[f"pred_{target}"] = float(pred_te[local, j])
                prediction_rows.append(rec)

    folds, preds, audit = pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows), pd.DataFrame(audit_rows)
    test = folds[folds["split"].eq("test")]
    metric_cols = [c for c in folds if c.endswith(("_R2", "_MAE", "_RMSE"))]
    summary = test.groupby(["method", "model"], sort=False)[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index().sort_values(["method", "macro_R2_mean"], ascending=[True, False])
    long_rows = []
    for (method, model), g in test.groupby(["method", "model"], sort=False):
        for target in TARGETS:
            for metric in ("MAE", "RMSE", "R2"):
                values = g[f"{target}_{metric}"].astype(float)
                long_rows.append({"method": method, "model": model, "target": target, "metric": metric,
                                  "mean": values.mean(), "std": values.std(ddof=1),
                                  "n_test_folds": values.notna().sum()})
    long_summary = pd.DataFrame(long_rows)
    # Do not interpret foldwise LOPPO R2: single-observation test folds have undefined R2.
    # Use pooled OOF R2/MAE/RMSE across all held-out groups instead.
    # Pooled OOF metrics: in each repeat, concatenate the mutually exclusive test-fold predictions.
    # Avoid treating repeated observations as independent samples.
    pooled_rows = []
    for (method, repeat, model), g in preds.groupby(["method", "repeat", "model"], sort=False):
        true = g[[f"true_{t}" for t in TARGETS]].to_numpy(float)
        pred = g[[f"pred_{t}" for t in TARGETS]].to_numpy(float)
        r = {"method": method, "repeat": repeat, "model": model, "n": len(g)}
        r.update(metrics(true, pred)); pooled_rows.append(r)
    pooled = pd.DataFrame(pooled_rows)
    pooled_summary = pooled.groupby(["method", "model"], sort=False)[metric_cols].agg(["mean", "std"])
    pooled_summary.columns = [f"{a}_{b}" for a, b in pooled_summary.columns]
    pooled_summary = pooled_summary.reset_index()
    # Origin-ready long-format table based on complete held-out predictions.
    origin_rows = []
    for _, row in pooled_summary.iterrows():
        for target in TARGETS:
            for metric in ("R2", "MAE", "RMSE"):
                col = f"{target}_{metric}"
                origin_rows.append({"method":row["method"], "model":row["model"],
                                    "target":target, "metric":metric,
                                    "mean":row.get(col+"_mean", np.nan),
                                    "std":row.get(col+"_std", np.nan)})
    origin_table = pd.DataFrame(origin_rows)
    # LOPPO audit: one row per held-out precursor pair, listing errors and sample count.
    per_pair = []
    for (method, model, pair), g in preds.groupby(["method", "model", "precursor_pair_group"], sort=False):
        if method != "leave_one_pair_out": continue
        rec = {"method":method, "model":model, "precursor_pair_group":pair, "n":len(g)}
        for target in TARGETS:
            diff = g[f"pred_{target}"].to_numpy() - g[f"true_{target}"].to_numpy()
            rec[f"{target}_MAE"] = float(np.mean(np.abs(diff)))
            rec[f"{target}_bias"] = float(np.mean(diff))
        per_pair.append(rec)
    for name, df in {"origin_pooled_oof_metrics_long.csv": origin_table,
                     "leave_one_pair_errors.csv": pd.DataFrame(per_pair),
                     "model_fold_metrics.csv": folds, "model_summary_mean_std.csv": summary,
                     "target_metric_summary_long.csv": long_summary, "oof_predictions.csv": preds,
                     "cv_split_audit.csv": audit, "pooled_oof_by_repeat.csv": pooled,
                     "pooled_oof_summary.csv": pooled_summary}.items():
        df.to_csv(output / name, index=False, encoding="utf-8-sig")
    return folds, summary, long_summary, preds, pooled_summary


def plot_summary(summary: pd.DataFrame, output: Path, dpi: int):
    """Plot same fixed model families across validation methods, with no retuning."""
    # LOPPO foldwise R2 is often undefined, so plotting pooled OOF R2 is correct.
    methods = list(summary["method"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, target in zip(axes.flat, ["Ex_max_nm", "Em_max_nm", "QY_%", "macro"]):
        metric = f"{target}_R2_mean"; err = f"{target}_R2_std"
        pos = np.arange(5); bar_width = 0.8 / len(methods)
        names = ["FPNN", "CatBoost", "0.3FPNN-0.7CatBoost", "0.5FPNN-0.5CatBoost", "0.7FPNN-0.3CatBoost"]
        for i, method in enumerate(methods):
            z = summary.set_index(["method", "model"]).reindex([(method, m) for m in names])
            ax.bar(pos - 0.4 + (i + 0.5) * bar_width, z[metric], bar_width, yerr=z[err],
                   label=method, capsize=2, alpha=0.85)
        ax.set_xticks(pos); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.7); ax.set_title("Macro-average" if target == "macro" else target)
        ax.set_ylabel(r"Test $R^2$ (fold mean ± SD)"); ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fixed models: random / precursor-pair grouped / leave-one-pair-out", weight="bold")
    fig.savefig(output / "Fig_split_method_comparison.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output / "Fig_split_method_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_pooled_summary(pooled_summary: pd.DataFrame, output: Path, dpi: int):
    """Main three-way comparison: pooled OOF across held-out samples, not foldwise R2."""
    methods = [m for m in ("random", "pair_grouped", "leave_one_pair_out")
               if m in set(pooled_summary["method"])]
    models = ["FPNN", "CatBoost", "0.3FPNN-0.7CatBoost",
              "0.5FPNN-0.5CatBoost", "0.7FPNN-0.3CatBoost"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, target in zip(axes.flat, ["Ex_max_nm", "Em_max_nm", "QY_%", "macro"]):
        metric = f"{target}_R2_mean"
        err = f"{target}_R2_std"
        w = 0.8 / len(methods)
        for k, method in enumerate(methods):
            vals = pooled_summary.set_index(["method", "model"]).reindex(
                [(method, m) for m in models])
            heights = vals[metric].to_numpy(dtype=float)
            yerr = np.nan_to_num(vals[err].to_numpy(dtype=float), nan=0)
            ax.bar(np.arange(len(models)) - 0.4 + (k+0.5)*w, heights, w,
                   yerr=yerr, capsize=2, label=method)
        ax.axhline(0, linewidth=0.8, color="black")
        ax.set_xticks(range(len(models)), models, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Pooled out-of-fold R² (mean ± SD across repeats)")
        ax.set_title("Macro-average" if target == "macro" else target)
        ax.grid(axis="y", alpha=0.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Random vs grouped vs leave-one-pair-out: held-out prediction R²",
                 weight="bold")
    fig.savefig(output / "Fig_three_split_pooled_OOF_R2.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output / "Fig_three_split_pooled_OOF_R2.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    cfg = Cfg()
    set_seed(cfg.cv_seed)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data = load_success1_data(cfg)
    _, feature_lists, selected = generate_selected_features(data, cfg)

    smiles = pd.read_excel(cfg.smiles_file)
    featurizer = CarbonDotFeaturizer(smiles, get_solvent_polarity_data())
    dataset = FPProcessDataset(data, featurizer)
    if len(dataset) != len(data):
        raise ValueError("FPNN and generated CatBoost features are not aligned.")

    pd.DataFrame({
        ROW_ID_COL: data[ROW_ID_COL], "success": data["success"],
        **{target: data[target] for target in TARGETS},
    }).to_csv(output / "success1_rows_used.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"target": t, "n_features": len(feature_lists[t]),
         "features": ";".join(feature_lists[t])}
        for t in TARGETS
    ]).to_csv(output / "selected_feature_manifest.csv", index=False, encoding="utf-8-sig")

    fpnn = FPProcessNN()
    pd.DataFrame([{
        "model": "FPProcessNN_from_02",
        "fingerprint_input_dim": 679,
        "fingerprint_hidden_dim": 128,
        "fingerprint_latent_dim": 32,
        "process_input_dim": 14,
        "process_output_dim": 16,
        "fusion_dim": 64,
        "output_dim": 3,
        "dropout_fp": 0.30,
        "dropout_process": 0.10,
        "ensemble_weight_fpnn": 0.50,
        "ensemble_weight_catboost": 0.50,
        "trainable_parameters": sum(
            p.numel() for p in fpnn.parameters() if p.requires_grad
        ),
    }]).to_csv(output / "model_architecture_manifest.csv", index=False, encoding="utf-8-sig")

    folds, summary, long_summary, _, pooled_summary = run_cv(data, dataset, selected, cfg, output)
    # The old fold-mean plot is omitted for LOPPO: fold R² is undefined for singletons.
    plot_pooled_summary(pooled_summary, output, cfg.dpi)
    with open(output / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), "interpretation": "Fixed preselected features/architecture/weights; results do not constitute independent nested selection evaluation."}, f, ensure_ascii=False, indent=2)

    print("\nCompleted.")
    print(summary[["method", "model", "macro_R2_mean", "macro_R2_std"]].to_string(index=False))
    print("\nPer-repeat pooled OOF summary:")
    print(pooled_summary[["method", "model", "macro_R2_mean", "macro_R2_std"]].to_string(index=False))
    print("NOTE: Feature lists, architecture and weights are PRESELECTED. This is a fixed-pipeline split comparison, not nested model selection.")
    print("LOPPO pooled R2 uses all 145 held-out predictions; singleton-pair fold R2 is undefined.")
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()

