# 02_train_fp_catboost_cycle.py
# -*- coding: utf-8 -*-
"""
Closed-loop Step 2:
Retrain FPProcessNN + CatBoost ensemble on the expanded training data.

Run:
    python 02_train_fp_catboost_cycle.py

Before running:
    Set Cfg.run_dir to the timestamped folder created by script 01.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, CatBoostClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model.model import data_process, CarbonDotFeaturizer

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski

TARGET_NAMES = ["Ex_max_nm", "Em_max_nm", "QY_%"]

# Morgan fingerprint
N_BITS = 256
RADIUS = 2

FEATURE_LIST_FILES = {
    "Ex_max_nm": "./cd_catboost_feature_selection/Ex_max_nm_best_top20_feature_list.csv",
    "Em_max_nm": "./cd_catboost_feature_selection/Em_max_nm_best_top35_feature_list.csv",
    "QY_%": "./cd_catboost_feature_selection/QY_%_best_top35_feature_list.csv",
}

FORBIDDEN_FEATURES = {
    "Ex_max_nm", "Em_max_nm", "QY_%",
    "stoke_shift", "stoke_shift_calc_check", "Stokes_shift_nm",
}


@dataclass
class Cfg:
    cycle: int = 18
    run_dir: str = "./closed_loop_runs/cycle_18_20261015_190319"
    expanded_training_file: str = ""  # optional override
    smiles_path: str = "./data/compound_smiles.xlsx"

    seeds: tuple = (42, 2024, 3407, 777, 1024)

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

    # Success classifier
    # If True, optical-property regressors are trained only on success=1 samples.
    # The success classifier itself is trained on both success=0 and success=1 samples.
    property_train_success_only: bool = True
    success_positive_threshold: float = 0.5
    success_feature_file_name: str = "success_classifier_feature_list.csv"

    # External selected P_success feature list from independent feature selection.
    # If this file exists, success classifier will use only these selected features.
    # The list is copied to <run_dir>/data/success_classifier_feature_list.csv
    # so that 03_ax_bo_search_cycle_success.py uses the same features.
    use_external_success_feature_list: bool = True
    external_success_feature_list_path: str = "./success_feature_selection_independent/success_classifier_feature_list_selected.csv"

    random_seed: int = 42


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_excel(p) if p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(p)

def norm_name(x: str) -> str:
    return str(x).strip().lower()


def safe_div(a, b, eps=1e-8):
    return a / (b + eps)


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if str(p).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(p)
    return pd.read_csv(p)


def norm_name(x: str) -> str:
    return str(x).strip().lower()


def safe_div(a, b, eps=1e-8):
    return a / (b + eps)


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if str(p).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(p)
    return pd.read_csv(p)


def mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None
    return Chem.MolFromSmiles(str(smiles))


def element_count(mol, atomic_num: int) -> int:
    return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == atomic_num)


def substructure_count(mol, smarts: str) -> int:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def has_substructure(mol, smarts: str) -> int:
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return int(mol.HasSubstructMatch(patt))


FG_SMARTS = {
    # oxygen-containing
    "fg_hydroxyl": "[OX2H]",
    "fg_phenol": "c[OX2H]",
    "fg_carboxylic_acid": "[CX3](=O)[OX2H1]",
    "fg_ester": "[CX3](=O)[OX2][#6]",
    "fg_aldehyde": "[CX3H1](=O)[#6]",
    "fg_ketone": "[#6][CX3](=O)[#6]",
    "fg_carbonyl_any": "[CX3]=[OX1]",

    # nitrogen-containing
    "fg_primary_amine": "[NX3;H2][#6]",
    "fg_secondary_amine": "[NX3;H1]([#6])[#6]",
    "fg_tertiary_amine": "[NX3;H0]([#6])([#6])[#6]",
    "fg_amide": "[NX3][CX3](=[OX1])[#6]",
    "fg_imide_or_urea_like": "[NX3][CX3](=[OX1])[NX3]",
    "fg_nitro": "[$([NX3](=O)=O),$([N+](=O)[O-])]",
    "fg_nitrile": "[CX2]#N",

    # sulfur / phosphorus
    "fg_thiol": "[SX2H]",
    "fg_sulfide": "[#16X2][#6]",
    "fg_sulfonic_acid": "[SX4](=O)(=O)[OX2H]",
    "fg_phosphate_like": "P(=O)(O)(O)",

    # aromatic / heterocycle
    "fg_aromatic_ring": "a1aaaaa1",
    "fg_heteroaromatic": "[a;r][n,o,s]",
}


def calc_mol_features(mol, prefix: str):
    if mol is None:
        raise ValueError(f"{prefix}: RDKit failed to parse SMILES")

    feats = {}

    # basic descriptors
    feats[f"{prefix}MolWt"] = Descriptors.MolWt(mol)
    feats[f"{prefix}ExactMolWt"] = Descriptors.ExactMolWt(mol)
    feats[f"{prefix}HeavyAtomCount"] = Descriptors.HeavyAtomCount(mol)
    feats[f"{prefix}NumValenceElectrons"] = Descriptors.NumValenceElectrons(mol)
    feats[f"{prefix}NumAromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    feats[f"{prefix}NumAliphaticRings"] = rdMolDescriptors.CalcNumAliphaticRings(mol)
    feats[f"{prefix}NumSaturatedRings"] = rdMolDescriptors.CalcNumSaturatedRings(mol)
    feats[f"{prefix}NumHAcceptors"] = Lipinski.NumHAcceptors(mol)
    feats[f"{prefix}NumHDonors"] = Lipinski.NumHDonors(mol)
    feats[f"{prefix}NumRotatableBonds"] = Lipinski.NumRotatableBonds(mol)
    feats[f"{prefix}TPSA"] = rdMolDescriptors.CalcTPSA(mol)
    feats[f"{prefix}MolLogP"] = Crippen.MolLogP(mol)
    feats[f"{prefix}FractionCSP3"] = rdMolDescriptors.CalcFractionCSP3(mol)
    feats[f"{prefix}FormalCharge"] = Chem.GetFormalCharge(mol)
    feats[f"{prefix}RingCount"] = rdMolDescriptors.CalcNumRings(mol)
    feats[f"{prefix}HeteroAtomCount"] = rdMolDescriptors.CalcNumHeteroatoms(mol)
    feats[f"{prefix}NumAtoms"] = mol.GetNumAtoms()

    # element counts
    c_count = element_count(mol, 6)
    h_count = sum(atom.GetTotalNumHs(includeNeighbors=True) for atom in mol.GetAtoms())
    n_count = element_count(mol, 7)
    o_count = element_count(mol, 8)
    s_count = element_count(mol, 16)
    p_count = element_count(mol, 15)

    feats[f"{prefix}C_count"] = c_count
    feats[f"{prefix}H_count"] = h_count
    feats[f"{prefix}N_count"] = n_count
    feats[f"{prefix}O_count"] = o_count
    feats[f"{prefix}S_count"] = s_count
    feats[f"{prefix}P_count"] = p_count

    feats[f"{prefix}N_to_C"] = safe_div(n_count, c_count)
    feats[f"{prefix}O_to_C"] = safe_div(o_count, c_count)
    feats[f"{prefix}S_to_C"] = safe_div(s_count, c_count)
    feats[f"{prefix}P_to_C"] = safe_div(p_count, c_count)
    feats[f"{prefix}Hetero_to_C"] = safe_div(n_count + o_count + s_count + p_count, c_count)

    # functional groups
    for key, smarts in FG_SMARTS.items():
        feats[f"{prefix}{key}_count"] = substructure_count(mol, smarts)
        feats[f"{prefix}{key}_has"] = has_substructure(mol, smarts)

    return feats


def calc_fp_bits(mol, prefix: str, n_bits=N_BITS, radius=RADIUS):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    out = {f"{prefix}fp_{i}": int(arr[i]) for i in range(n_bits)}
    return fp, out


def solvent_feature_dict():
    return {
        "water": {
            "solv_is_water": 1, "solv_is_alcohol": 0, "solv_is_amide_solvent": 0,
            "solv_is_acid": 0, "solv_is_strong_acid": 0, "solv_is_protic": 1,
            "solv_has_n": 0, "solv_dielectric": 78.5, "solv_bp_C": 100.0,
            "solv_polarity_index": 9.0, "solv_relative_acidity_level": 0,
        },
        "ethanol": {
            "solv_is_water": 0, "solv_is_alcohol": 1, "solv_is_amide_solvent": 0,
            "solv_is_acid": 0, "solv_is_strong_acid": 0, "solv_is_protic": 1,
            "solv_has_n": 0, "solv_dielectric": 24.3, "solv_bp_C": 78.4,
            "solv_polarity_index": 5.2, "solv_relative_acidity_level": 0,
        },
        "dimethylformamide": {
            "solv_is_water": 0, "solv_is_alcohol": 0, "solv_is_amide_solvent": 1,
            "solv_is_acid": 0, "solv_is_strong_acid": 0, "solv_is_protic": 0,
            "solv_has_n": 1, "solv_dielectric": 36.7, "solv_bp_C": 153.0,
            "solv_polarity_index": 6.4, "solv_relative_acidity_level": 0,
        },
        "phosphoric acid": {
            "solv_is_water": 0, "solv_is_alcohol": 0, "solv_is_amide_solvent": 0,
            "solv_is_acid": 1, "solv_is_strong_acid": 1, "solv_is_protic": 1,
            "solv_has_n": 0, "solv_dielectric": 61.0, "solv_bp_C": 158.0,
            "solv_polarity_index": 8.0, "solv_relative_acidity_level": 2,
        },
        "sulfuric acid": {
            "solv_is_water": 0, "solv_is_alcohol": 0, "solv_is_amide_solvent": 0,
            "solv_is_acid": 1, "solv_is_strong_acid": 1, "solv_is_protic": 1,
            "solv_has_n": 0, "solv_dielectric": 100.0, "solv_bp_C": 337.0,
            "solv_polarity_index": 9.5, "solv_relative_acidity_level": 3,
        },
    }


def build_compound_feature_table(df_smiles: pd.DataFrame, n_bits=N_BITS):
    rows = []
    fp_objs = {}

    for _, row in df_smiles.iterrows():
        name = row["compound_name"]
        smiles = row["SMILES"]
        mol = mol_from_smiles(smiles)

        if mol is None:
            warnings.warn(f"Cannot parse SMILES for {name}: {smiles}")
            continue

        feats = {
            "compound_name": name,
            "compound_key": norm_name(name),
            "SMILES": smiles,
        }
        feats.update(calc_mol_features(mol, prefix=""))

        fp, fp_bits = calc_fp_bits(mol, prefix="", n_bits=n_bits)
        feats.update(fp_bits)

        rows.append(feats)
        fp_objs[norm_name(name)] = fp

    feat_df = pd.DataFrame(rows)
    return feat_df, fp_objs


def standardize_candidate_columns(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make candidate_pooling.csv compatible with the original feature-engineering code.

    Required original-style aliases:
        Mass1(g)
        Mass2(g)
        reagent_1_mL
    """
    out = candidate_df.copy()

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
            out["Mass1(g)"] = out["mass_1_mg"] / 1000.0
        else:
            raise ValueError("Candidate file must contain Mass1(g), mass_1_g, or mass_1_mg.")

    if "Mass2(g)" not in out.columns:
        if "mass_2_g" in out.columns:
            out["Mass2(g)"] = out["mass_2_g"]
        elif "mass_2_mg" in out.columns:
            out["Mass2(g)"] = out["mass_2_mg"] / 1000.0
        else:
            raise ValueError("Candidate file must contain Mass2(g), mass_2_g, or mass_2_mg.")

    if "reagent_1_mL" not in out.columns:
        out["reagent_1_mL"] = DEFAULT_REAGENT_1_ML

    required = [
        "precursor_1", "precursor_2", "reagent_1",
        "pre_1_mmol", "pre_2_mmol",
        "Mass1(g)", "Mass2(g)",
        "reagent_1_mL", "temperature", "time_h",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Candidate file missing required columns after standardization: {missing}")

    return out


def build_dataset_features(exp_df: pd.DataFrame, compound_feat_df: pd.DataFrame, fp_objs: dict, has_targets: bool):
    """
    Build feature table for training data or candidate data.

    If has_targets=False, target columns are not required and stoke_shift will not be generated
    from measured Ex/Em.
    """
    exp = standardize_candidate_columns(exp_df)

    exp["precursor_1_key"] = exp["precursor_1"].map(norm_name)
    exp["precursor_2_key"] = exp["precursor_2"].map(norm_name)
    exp["reagent_1_key"] = exp["reagent_1"].map(norm_name)

    feat = compound_feat_df.copy()
    feat1 = feat.add_prefix("p1_")
    feat2 = feat.add_prefix("p2_")

    out = exp.merge(
        feat1,
        left_on="precursor_1_key",
        right_on="p1_compound_key",
        how="left"
    ).merge(
        feat2,
        left_on="precursor_2_key",
        right_on="p2_compound_key",
        how="left"
    )

    missing1 = out["p1_SMILES"].isna().sum()
    missing2 = out["p2_SMILES"].isna().sum()
    if missing1 or missing2:
        missing_p1 = sorted(out.loc[out["p1_SMILES"].isna(), "precursor_1"].dropna().unique().tolist())
        missing_p2 = sorted(out.loc[out["p2_SMILES"].isna(), "precursor_2"].dropna().unique().tolist())
        raise ValueError(
            f"Missing merged SMILES/features: precursor_1 missing={missing1}, precursor_2 missing={missing2}\n"
            f"Missing precursor_1 names: {missing_p1[:20]}\n"
            f"Missing precursor_2 names: {missing_p2[:20]}"
        )

    # solvent engineered features
    solv_map = solvent_feature_dict()
    solvent_rows = []
    for s in out["reagent_1_key"]:
        if s not in solv_map:
            raise KeyError(f"Solvent '{s}' not found in solvent feature dict.")
        solvent_rows.append(solv_map[s])

    solv_df = pd.DataFrame(solvent_rows, index=out.index)
    out = pd.concat([out, solv_df], axis=1)

    # process / stoichiometry features
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

    # Alias for compatibility with selected CatBoost feature lists.
    # The value represents solvent/reagent volume normalized by total precursor mass, i.e., mL/g.
    # Some selected feature files use the name "reagent_to_precursor_vol_mass_ratio".
    out["reagent_to_precursor_vol_mass_ratio"] = out["reagent_to_precursor_mass_ratio"]

    out["temp_time"] = out["temperature"] * out["time_h"]
    out["temp_over_time"] = safe_div(out["temperature"], out["time_h"])
    out["same_precursor_flag"] = (out["precursor_1_key"] == out["precursor_2_key"]).astype(int)

    # pairwise descriptor interactions
    continuous_base = [
        "MolWt", "ExactMolWt", "HeavyAtomCount", "NumValenceElectrons",
        "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
        "NumHAcceptors", "NumHDonors", "NumRotatableBonds", "TPSA", "MolLogP",
        "FractionCSP3", "FormalCharge", "RingCount", "HeteroAtomCount", "NumAtoms",
        "C_count", "H_count", "N_count", "O_count", "S_count", "P_count",
        "N_to_C", "O_to_C", "S_to_C", "P_to_C", "Hetero_to_C"
    ]

    fg_base = []
    for k in FG_SMARTS.keys():
        fg_base.append(f"{k}_count")
        fg_base.append(f"{k}_has")

    pair_base = continuous_base + fg_base

    for col in pair_base:
        a = out[f"p1_{col}"]
        b = out[f"p2_{col}"]
        out[f"pair_sum_{col}"] = a + b
        out[f"pair_absdiff_{col}"] = (a - b).abs()
        out[f"pair_prod_{col}"] = a * b

    ratio_cols = ["MolWt", "N_count", "O_count", "S_count", "P_count", "TPSA", "MolLogP"]
    for col in ratio_cols:
        a = out[f"p1_{col}"]
        b = out[f"p2_{col}"]
        out[f"pair_ratio12_{col}"] = safe_div(a, b)
        out[f"pair_ratio21_{col}"] = safe_div(b, a)

    # pair-level chemistry cues
    out["pair_total_N"] = out["p1_N_count"] + out["p2_N_count"]
    out["pair_total_O"] = out["p1_O_count"] + out["p2_O_count"]
    out["pair_total_S"] = out["p1_S_count"] + out["p2_S_count"]
    out["pair_total_P"] = out["p1_P_count"] + out["p2_P_count"]
    out["pair_total_C"] = out["p1_C_count"] + out["p2_C_count"]

    out["pair_N_to_C"] = safe_div(out["pair_total_N"], out["pair_total_C"])
    out["pair_O_to_C"] = safe_div(out["pair_total_O"], out["pair_total_C"])
    out["pair_S_to_C"] = safe_div(out["pair_total_S"], out["pair_total_C"])
    out["pair_P_to_C"] = safe_div(out["pair_total_P"], out["pair_total_C"])

    out["pair_has_N_and_O"] = ((out["pair_total_N"] > 0) & (out["pair_total_O"] > 0)).astype(int)
    out["pair_has_N_and_S"] = ((out["pair_total_N"] > 0) & (out["pair_total_S"] > 0)).astype(int)
    out["pair_has_N_and_P"] = ((out["pair_total_N"] > 0) & (out["pair_total_P"] > 0)).astype(int)
    out["pair_both_aromatic"] = ((out["p1_fg_aromatic_ring_has"] > 0) & (out["p2_fg_aromatic_ring_has"] > 0)).astype(int)
    out["pair_one_aromatic_one_nonaromatic"] = ((out["p1_fg_aromatic_ring_has"] > 0) ^ (out["p2_fg_aromatic_ring_has"] > 0)).astype(int)
    out["pair_amine_plus_acid"] = (
        ((out["p1_fg_primary_amine_has"] + out["p1_fg_secondary_amine_has"] + out["p1_fg_tertiary_amine_has"]) > 0) &
        (out["p2_fg_carboxylic_acid_has"] > 0)
    ).astype(int) + (
        ((out["p2_fg_primary_amine_has"] + out["p2_fg_secondary_amine_has"] + out["p2_fg_tertiary_amine_has"]) > 0) &
        (out["p1_fg_carboxylic_acid_has"] > 0)
    ).astype(int)

    # fingerprint pair features
    tanimoto_list = []
    inter_counts = []
    union_counts = []
    xor_counts = []

    for _, r in out.iterrows():
        fp1 = fp_objs[r["precursor_1_key"]]
        fp2 = fp_objs[r["precursor_2_key"]]

        tanimoto_list.append(DataStructs.TanimotoSimilarity(fp1, fp2))

        arr1 = np.array([int(x) for x in fp1.ToBitString()], dtype=np.int8)
        arr2 = np.array([int(x) for x in fp2.ToBitString()], dtype=np.int8)

        inter_counts.append(int(np.logical_and(arr1, arr2).sum()))
        union_counts.append(int(np.logical_or(arr1, arr2).sum()))
        xor_counts.append(int(np.logical_xor(arr1, arr2).sum()))

    out["fp_tanimoto"] = tanimoto_list
    out["fp_intersection_count"] = inter_counts
    out["fp_union_count"] = union_counts
    out["fp_xor_count"] = xor_counts

    # optional: keep original fp bits for both precursors
    fp_cols = [c for c in compound_feat_df.columns if c.startswith("fp_")]
    for c in fp_cols:
        out[f"p1_{c}"] = out[f"p1_{c}"].astype(np.int8)
        out[f"p2_{c}"] = out[f"p2_{c}"].astype(np.int8)

    # Add QY-oriented process-structure interaction features.
    # These are based only on precursor descriptors, solvent descriptors, and process variables.
    out = add_qy_extra_features(out)

    if has_targets:
        if "stoke_shift" not in out.columns:
            out["stoke_shift"] = out["Em_max_nm"] - out["Ex_max_nm"]
        else:
            out["stoke_shift_calc_check"] = out["Em_max_nm"] - out["Ex_max_nm"]

    return out



def add_qy_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add QY-oriented engineered interaction features.

    Memory-safe version:
    - does NOT call df.copy()
    - modifies the input dataframe in place
    - casts newly generated numeric columns to float32 where possible

    These qy_* columns are not derived from measured QY_%.
    They are process-structure interaction descriptors designed for QY modeling.
    """
    out = df  # important: avoid copying a 1,176,000 × ~900 dataframe

    def col(name, default=0.0):
        if name in out.columns:
            return out[name]
        return default

    pair_total_hetero = (
        col("pair_total_N")
        + col("pair_total_O")
        + col("pair_total_S")
        + col("pair_total_P")
    )

    # Use float32 to reduce memory pressure.
    out["qy_pair_total_hetero"] = pair_total_hetero.astype("float32")
    out["qy_pair_hetero_to_C"] = (
        pair_total_hetero / (col("pair_total_C") + 1e-8)
    ).astype("float32")

    out["qy_acidity_x_temp"] = (
        col("solv_relative_acidity_level") * col("temperature")
    ).astype("float32")
    out["qy_acidity_x_time"] = (
        col("solv_relative_acidity_level") * col("time_h")
    ).astype("float32")
    out["qy_acidity_x_temp_time"] = (
        col("solv_relative_acidity_level") * col("temp_time")
    ).astype("float32")
    out["qy_protic_x_temp_time"] = (
        col("solv_is_protic") * col("temp_time")
    ).astype("float32")

    out["qy_acid_x_pair_total_N"] = (
        col("solv_relative_acidity_level") * col("pair_total_N")
    ).astype("float32")
    out["qy_acid_x_pair_total_O"] = (
        col("solv_relative_acidity_level") * col("pair_total_O")
    ).astype("float32")
    out["qy_acid_x_pair_total_S"] = (
        col("solv_relative_acidity_level") * col("pair_total_S")
    ).astype("float32")
    out["qy_acid_x_pair_total_P"] = (
        col("solv_relative_acidity_level") * col("pair_total_P")
    ).astype("float32")

    out["qy_temp_x_pair_total_hetero"] = (
        col("temperature") * pair_total_hetero
    ).astype("float32")
    out["qy_time_x_pair_total_hetero"] = (
        col("time_h") * pair_total_hetero
    ).astype("float32")
    out["qy_temp_time_x_pair_total_hetero"] = (
        col("temp_time") * pair_total_hetero
    ).astype("float32")

    out["qy_acid_x_pair_N_to_C"] = (
        col("solv_relative_acidity_level") * col("pair_N_to_C")
    ).astype("float32")
    out["qy_acid_x_pair_O_to_C"] = (
        col("solv_relative_acidity_level") * col("pair_O_to_C")
    ).astype("float32")
    out["qy_acid_x_pair_S_to_C"] = (
        col("solv_relative_acidity_level") * col("pair_S_to_C")
    ).astype("float32")
    out["qy_acid_x_pair_P_to_C"] = (
        col("solv_relative_acidity_level") * col("pair_P_to_C")
    ).astype("float32")

    out["qy_acid_x_amine_plus_acid"] = (
        col("solv_relative_acidity_level") * col("pair_amine_plus_acid")
    ).astype("float32")
    out["qy_protic_x_amine_plus_acid"] = (
        col("solv_is_protic") * col("pair_amine_plus_acid")
    ).astype("float32")

    out["qy_temp_x_tanimoto"] = (
        col("temperature") * col("fp_tanimoto")
    ).astype("float32")
    out["qy_time_x_tanimoto"] = (
        col("time_h") * col("fp_tanimoto")
    ).astype("float32")
    out["qy_acid_x_tanimoto"] = (
        col("solv_relative_acidity_level") * col("fp_tanimoto")
    ).astype("float32")

    out["qy_reagent_ratio_x_temp"] = (
        col("reagent_to_precursor_vol_mass_ratio") * col("temperature")
    ).astype("float32")
    out["qy_reagent_ratio_x_time"] = (
        col("reagent_to_precursor_vol_mass_ratio") * col("time_h")
    ).astype("float32")
    out["qy_reagent_ratio_x_hetero"] = (
        col("reagent_to_precursor_vol_mass_ratio") * pair_total_hetero
    ).astype("float32")

    out["qy_mass_ratio_x_acidity"] = (
        col("mass_ratio_12") * col("solv_relative_acidity_level")
    ).astype("float32")
    out["qy_mmol_ratio_x_acidity"] = (
        col("pre_mmol_ratio_12") * col("solv_relative_acidity_level")
    ).astype("float32")

    return out


def get_selected_feature_columns(selected_feature_file: str, target_col: str) -> list[str]:
    df_sel = read_table(selected_feature_file)

    target_cols = {"Ex_max_nm", "Em_max_nm", "QY_%", "stoke_shift", "stoke_shift_calc_check"}
    meta_cols = {
        "precursor_1", "precursor_2", "reagent_1",
        "precursor_1_key", "precursor_2_key", "reagent_1_key",
        "p1_compound_name", "p1_compound_key", "p1_SMILES",
        "p2_compound_name", "p2_compound_key", "p2_SMILES",
        "compound_name", "compound_key", "SMILES",
    }

    selected_cols = [
        c for c in df_sel.columns
        if c not in target_cols and c not in meta_cols
    ]

    if len(selected_cols) == 0:
        raise ValueError(f"No selected feature columns found in {selected_feature_file}")

    return selected_cols




def make_unique_keep_order(cols: list[str]) -> list[str]:
    """
    Remove duplicated column names while preserving order.
    """
    seen = set()
    unique = []
    for c in cols:
        if c not in seen:
            unique.append(c)
            seen.add(c)
    return unique


def prepare_selected_feature_info():
    """
    Read selected feature files once and prepare target-specific output settings.
    """
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = {}

    for target, selected_path in SELECTED_FEATURE_FILES.items():
        selected_cols_raw = get_selected_feature_columns(selected_path, target_col=target)
        selected_cols = make_unique_keep_order(selected_cols_raw)

        out_path = out_dir / f"{target}_candidate_features.csv"

        # Remove old output file.
        if out_path.exists():
            out_path.unlink()

        info[target] = {
            "selected_path": selected_path,
            "selected_cols_raw": selected_cols_raw,
            "selected_cols": selected_cols,
            "out_path": out_path,
            "header_written": False,
            "n_rows": 0,
            "missing_cols_union": set(),
            "n_chunks": 0,
        }

    return info


def write_candidate_feature_chunk(candidate_feature_df: pd.DataFrame, selected_info: dict):
    """
    Write one candidate feature chunk to target-specific CatBoost candidate files.

    This function is memory-safe:
    - only slices current chunk;
    - only writes meta columns + selected columns;
    - does not save the full 900+ column candidate feature table.
    """
    if candidate_feature_df.columns.duplicated().any():
        dup_cols = candidate_feature_df.columns[candidate_feature_df.columns.duplicated()].tolist()
        warnings.warn(
            f"candidate_feature_df chunk contains duplicated columns. "
            f"Duplicated columns will be removed. First duplicates: {dup_cols[:20]}"
        )
        candidate_feature_df = candidate_feature_df.loc[:, ~candidate_feature_df.columns.duplicated()].copy()

    meta_keep = [
        "candidate_id",
        "precursor_1", "precursor_2", "reagent_1",
        "temperature", "time_h",
        "Mass1(g)", "Mass2(g)",
        "pre_1_mmol", "pre_2_mmol",
        "pre_mmol_sum", "pre1_mmol_frac", "pre2_mmol_frac",
    ]
    meta_keep = [c for c in meta_keep if c in candidate_feature_df.columns]
    meta_keep = make_unique_keep_order(meta_keep)

    for target, info in selected_info.items():
        selected_cols = info["selected_cols"]

        missing = [c for c in selected_cols if c not in candidate_feature_df.columns]
        if missing:
            info["missing_cols_union"].update(missing)
            for c in missing:
                candidate_feature_df[c] = 0.0

        out_cols = make_unique_keep_order(meta_keep + selected_cols)
        out_df = candidate_feature_df.loc[:, out_cols].copy()

        # Ensure selected feature columns are numeric.
        for c in selected_cols:
            if c not in out_df.columns:
                out_df[c] = 0.0

            col_obj = out_df.loc[:, c]
            if isinstance(col_obj, pd.DataFrame):
                warnings.warn(
                    f"{target}: column '{c}' is duplicated in out_df chunk. "
                    "Using the first occurrence."
                )
                out_df[c] = pd.to_numeric(col_obj.iloc[:, 0], errors="coerce").fillna(0.0)
            else:
                out_df[c] = pd.to_numeric(col_obj, errors="coerce").fillna(0.0)

        out_df.to_csv(
            info["out_path"],
            mode="a",
            header=not info["header_written"],
            index=False,
            encoding="utf-8-sig",
        )

        info["header_written"] = True
        info["n_rows"] += len(out_df)
        info["n_chunks"] += 1


def write_selected_feature_summary(selected_info: dict):
    """
    Save generation summary after all chunks are processed.
    """
    out_dir = Path(OUT_DIR)
    summary_rows = []

    for target, info in selected_info.items():
        missing = sorted(list(info["missing_cols_union"]))

        summary_rows.append({
            "target": target,
            "selected_feature_file": info["selected_path"],
            "output_file": str(info["out_path"]),
            "n_rows": info["n_rows"],
            "n_chunks": info["n_chunks"],
            "n_selected_features_raw": len(info["selected_cols_raw"]),
            "n_selected_features_unique": len(info["selected_cols"]),
            "n_duplicate_selected_removed": len(info["selected_cols_raw"]) - len(info["selected_cols"]),
            "n_missing_filled_zero": len(missing),
            "missing_features": ";".join(missing),
        })

        print(f"[{target}] saved: {info['out_path']}")
        print(f"  rows={info['n_rows']:,}")
        print(f"  chunks={info['n_chunks']}")
        print(f"  selected_features_raw={len(info['selected_cols_raw'])}")
        print(f"  selected_features_unique={len(info['selected_cols'])}")
        print(f"  missing_filled_zero={len(missing)}")
        if missing:
            print(f"  missing features: {missing[:20]}")

    pd.DataFrame(summary_rows).to_csv(
        out_dir / "candidate_feature_generation_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def iter_candidate_chunks(path: str, chunksize: int):
    """
    Yield candidate dataframe chunks.
    CSV supports streaming. Excel is loaded once and then split into chunks.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidate pool file not found: {p}")

    if str(p).lower().endswith(".csv"):
        for chunk in pd.read_csv(p, chunksize=chunksize):
            yield chunk
    else:
        # Excel cannot be streamed efficiently by pandas.
        # Use CSV for million-scale candidate pools.
        df = pd.read_excel(p)
        for start in range(0, len(df), chunksize):
            yield df.iloc[start:start + chunksize].copy()


FG_SMARTS = {
    # oxygen-containing
    "fg_hydroxyl": "[OX2H]",
    "fg_phenol": "c[OX2H]",
    "fg_carboxylic_acid": "[CX3](=O)[OX2H1]",
    "fg_ester": "[CX3](=O)[OX2][#6]",
    "fg_aldehyde": "[CX3H1](=O)[#6]",
    "fg_ketone": "[#6][CX3](=O)[#6]",
    "fg_carbonyl_any": "[CX3]=[OX1]",

    # nitrogen-containing
    "fg_primary_amine": "[NX3;H2][#6]",
    "fg_secondary_amine": "[NX3;H1]([#6])[#6]",
    "fg_tertiary_amine": "[NX3;H0]([#6])([#6])[#6]",
    "fg_amide": "[NX3][CX3](=[OX1])[#6]",
    "fg_imide_or_urea_like": "[NX3][CX3](=[OX1])[NX3]",
    "fg_nitro": "[$([NX3](=O)=O),$([N+](=O)[O-])]",
    "fg_nitrile": "[CX2]#N",

    # sulfur / phosphorus
    "fg_thiol": "[SX2H]",
    "fg_sulfide": "[#16X2][#6]",
    "fg_sulfonic_acid": "[SX4](=O)(=O)[OX2H]",
    "fg_phosphate_like": "P(=O)(O)(O)",

    # aromatic / heterocycle
    "fg_aromatic_ring": "a1aaaaa1",
    "fg_heteroaromatic": "[a;r][n,o,s]",
}


def calc_mol_features(mol, prefix: str):
    if mol is None:
        raise ValueError(f"{prefix}: RDKit failed to parse SMILES")

    feats = {}

    # basic descriptors
    feats[f"{prefix}MolWt"] = Descriptors.MolWt(mol)
    feats[f"{prefix}ExactMolWt"] = Descriptors.ExactMolWt(mol)
    feats[f"{prefix}HeavyAtomCount"] = Descriptors.HeavyAtomCount(mol)
    feats[f"{prefix}NumValenceElectrons"] = Descriptors.NumValenceElectrons(mol)
    feats[f"{prefix}NumAromaticRings"] = rdMolDescriptors.CalcNumAromaticRings(mol)
    feats[f"{prefix}NumAliphaticRings"] = rdMolDescriptors.CalcNumAliphaticRings(mol)
    feats[f"{prefix}NumSaturatedRings"] = rdMolDescriptors.CalcNumSaturatedRings(mol)
    feats[f"{prefix}NumHAcceptors"] = Lipinski.NumHAcceptors(mol)
    feats[f"{prefix}NumHDonors"] = Lipinski.NumHDonors(mol)
    feats[f"{prefix}NumRotatableBonds"] = Lipinski.NumRotatableBonds(mol)
    feats[f"{prefix}TPSA"] = rdMolDescriptors.CalcTPSA(mol)
    feats[f"{prefix}MolLogP"] = Crippen.MolLogP(mol)
    feats[f"{prefix}FractionCSP3"] = rdMolDescriptors.CalcFractionCSP3(mol)
    feats[f"{prefix}FormalCharge"] = Chem.GetFormalCharge(mol)
    feats[f"{prefix}RingCount"] = rdMolDescriptors.CalcNumRings(mol)
    feats[f"{prefix}HeteroAtomCount"] = rdMolDescriptors.CalcNumHeteroatoms(mol)
    feats[f"{prefix}NumAtoms"] = mol.GetNumAtoms()

    # element counts
    c_count = element_count(mol, 6)
    h_count = sum(atom.GetTotalNumHs(includeNeighbors=True) for atom in mol.GetAtoms())
    n_count = element_count(mol, 7)
    o_count = element_count(mol, 8)
    s_count = element_count(mol, 16)
    p_count = element_count(mol, 15)

    feats[f"{prefix}C_count"] = c_count
    feats[f"{prefix}H_count"] = h_count
    feats[f"{prefix}N_count"] = n_count
    feats[f"{prefix}O_count"] = o_count
    feats[f"{prefix}S_count"] = s_count
    feats[f"{prefix}P_count"] = p_count

    feats[f"{prefix}N_to_C"] = safe_div(n_count, c_count)
    feats[f"{prefix}O_to_C"] = safe_div(o_count, c_count)
    feats[f"{prefix}S_to_C"] = safe_div(s_count, c_count)
    feats[f"{prefix}P_to_C"] = safe_div(p_count, c_count)
    feats[f"{prefix}Hetero_to_C"] = safe_div(n_count + o_count + s_count + p_count, c_count)

    # functional groups
    for key, smarts in FG_SMARTS.items():
        feats[f"{prefix}{key}_count"] = substructure_count(mol, smarts)
        feats[f"{prefix}{key}_has"] = has_substructure(mol, smarts)

    return feats


def calc_fp_bits(mol, prefix: str, n_bits=N_BITS, radius=RADIUS):
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    out = {f"{prefix}fp_{i}": int(arr[i]) for i in range(n_bits)}
    return fp, out


def solvent_feature_dict():
    return {
        "water": {
            "solv_is_water": 1, "solv_is_alcohol": 0, "solv_is_amide_solvent": 0,
            "solv_is_acid": 0, "solv_is_strong_acid": 0, "solv_is_protic": 1,
            "solv_has_n": 0, "solv_dielectric": 78.5, "solv_bp_C": 100.0,
            "solv_polarity_index": 9.0, "solv_relative_acidity_level": 0,
        },
        "ethanol": {
            "solv_is_water": 0, "solv_is_alcohol": 1, "solv_is_amide_solvent": 0,
            "solv_is_acid": 0, "solv_is_strong_acid": 0, "solv_is_protic": 1,
            "solv_has_n": 0, "solv_dielectric": 24.3, "solv_bp_C": 78.4,
            "solv_polarity_index": 5.2, "solv_relative_acidity_level": 0,
        },
        "dimethylformamide": {
            "solv_is_water": 0, "solv_is_alcohol": 0, "solv_is_amide_solvent": 1,
            "solv_is_acid": 0, "solv_is_strong_acid": 0, "solv_is_protic": 0,
            "solv_has_n": 1, "solv_dielectric": 36.7, "solv_bp_C": 153.0,
            "solv_polarity_index": 6.4, "solv_relative_acidity_level": 0,
        },
        "phosphoric acid": {
            "solv_is_water": 0, "solv_is_alcohol": 0, "solv_is_amide_solvent": 0,
            "solv_is_acid": 1, "solv_is_strong_acid": 1, "solv_is_protic": 1,
            "solv_has_n": 0, "solv_dielectric": 61.0, "solv_bp_C": 158.0,
            "solv_polarity_index": 8.0, "solv_relative_acidity_level": 2,
        },
        "sulfuric acid": {
            "solv_is_water": 0, "solv_is_alcohol": 0, "solv_is_amide_solvent": 0,
            "solv_is_acid": 1, "solv_is_strong_acid": 1, "solv_is_protic": 1,
            "solv_has_n": 0, "solv_dielectric": 100.0, "solv_bp_C": 337.0,
            "solv_polarity_index": 9.5, "solv_relative_acidity_level": 3,
        },
    }

def standardize_candidate_columns(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make candidate_pooling.csv compatible with the original feature-engineering code.

    Required original-style aliases:
        Mass1(g)
        Mass2(g)
        reagent_1_mL
    """
    out = candidate_df.copy()

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
            out["Mass1(g)"] = out["mass_1_mg"] / 1000.0
        else:
            raise ValueError("Candidate file must contain Mass1(g), mass_1_g, or mass_1_mg.")

    if "Mass2(g)" not in out.columns:
        if "mass_2_g" in out.columns:
            out["Mass2(g)"] = out["mass_2_g"]
        elif "mass_2_mg" in out.columns:
            out["Mass2(g)"] = out["mass_2_mg"] / 1000.0
        else:
            raise ValueError("Candidate file must contain Mass2(g), mass_2_g, or mass_2_mg.")

    if "reagent_1_mL" not in out.columns:
        out["reagent_1_mL"] = DEFAULT_REAGENT_1_ML

    required = [
        "precursor_1", "precursor_2", "reagent_1",
        "pre_1_mmol", "pre_2_mmol",
        "Mass1(g)", "Mass2(g)",
        "reagent_1_mL", "temperature", "time_h",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Candidate file missing required columns after standardization: {missing}")

    return out


def get_solvent_polarity_data():
    return {
        "water": {"dielectric_constant": 80.1, "pi_star": 1.09, "HBD": 1.17, "HBA": 0.47, "ET30": 63.1, "pKa": 15.7, "ORP_mV": 300, "density_g_cm3": 0.997, "specific_heat_J_gK": 4.18, "viscosity_mPa_s": 0.89, "boiling_point_C": 100},
        "dimethylformamide": {"dielectric_constant": 36.7, "pi_star": 0.88, "HBD": 0.00, "HBA": 0.69, "ET30": 43.2, "pKa": 29, "ORP_mV": 200, "density_g_cm3": 0.944, "specific_heat_J_gK": 2.47, "viscosity_mPa_s": 0.92, "boiling_point_C": 153},
        "ethanol": {"dielectric_constant": 24.6, "pi_star": 0.54, "HBD": 0.86, "HBA": 0.75, "ET30": 51.9, "pKa": 15.9, "ORP_mV": 250, "density_g_cm3": 0.789, "specific_heat_J_gK": 2.44, "viscosity_mPa_s": 1.07, "boiling_point_C": 78.37},
        "sulfuric acid": {"dielectric_constant": 83, "pi_star": 1.11, "HBD": 1.22, "HBA": 0.48, "ET30": 64.8, "pKa": 0.0, "ORP_mV": 800, "density_g_cm3": 1.05, "specific_heat_J_gK": 3.6, "viscosity_mPa_s": 1.2, "boiling_point_C": 100},
        "phosphoric acid": {"dielectric_constant": 82, "pi_star": 1.10, "HBD": 1.20, "HBA": 0.47, "ET30": 64.0, "pKa": 1.0, "ORP_mV": 700, "density_g_cm3": 1.03, "specific_heat_J_gK": 3.8, "viscosity_mPa_s": 1.1, "boiling_point_C": 100},
    }



class FPProcessDataset(Dataset):
    def __init__(self, df, featurizer, has_y=True):
        self.df = df.reset_index(drop=True)
        self.featurizer = featurizer
        self.has_y = has_y
        precursors = sorted(set(self.df["precursor_1"].astype(str)) | set(self.df["precursor_2"].astype(str)))
        self.fp_map = {p: torch.tensor(self.featurizer.fingerprints(p), dtype=torch.float32) for p in precursors}
        reagents = sorted(set(self.df["reagent_1"].astype(str)))
        self.solv_map = {r: self.featurizer.solvent_feat(r).float() for r in reagents}

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        fp1 = self.fp_map[str(r["precursor_1"])]
        fp2 = self.fp_map[str(r["precursor_2"])]
        s_feat = self.solv_map[str(r["reagent_1"])]
        r_feat = self.featurizer.reaction_feat(r["temperature"], r["time_h"]).float()
        pre1, pre2 = float(r["pre_1_mmol"]), float(r["pre_2_mmol"])
        ratio = pre1 / (pre1 + pre2) if (pre1 + pre2) > 0 else 0.5
        item = {"fps_1": fp1, "fps_2": fp2, "global_feat": torch.cat([s_feat, r_feat, torch.tensor([ratio])])}
        if self.has_y:
            item["y"] = torch.tensor([r["Ex_max_nm"], r["Em_max_nm"], r["QY_%"]], dtype=torch.float32)
        return item


def collate_fn(batch):
    out = {
        "fps_1": torch.stack([b["fps_1"] for b in batch]),
        "fps_2": torch.stack([b["fps_2"] for b in batch]),
        "global_feat": torch.stack([b["global_feat"] for b in batch]),
    }
    if "y" in batch[0]:
        out["y"] = torch.stack([b["y"] for b in batch])
    return out


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
        fusion = self.fusion_mlp(torch.cat([f1 + f2, torch.abs(f1 - f2)], dim=-1))
        proc = self.proc_encoder(global_feat)
        return self.head(torch.cat([fusion, proc], dim=-1))


def train_one_nn(seed, dataset, y_scaler, cfg, model_dir):
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FPProcessNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.nn_lr, weight_decay=cfg.nn_weight_decay)
    loader = DataLoader(dataset, batch_size=cfg.nn_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

    for ep in range(cfg.nn_epochs):
        losses = []
        model.train()
        for batch in loader:
            y_scaled = y_scaler.transform(batch["y"].numpy().astype(np.float32)).astype(np.float32)
            pred = model(batch["fps_1"].to(dev), batch["fps_2"].to(dev), batch["global_feat"].to(dev))
            loss = F.mse_loss(pred, torch.tensor(y_scaled, dtype=torch.float32, device=dev))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.nn_grad_clip)
            opt.step()
            losses.append(loss.item())
        if (ep + 1) % 50 == 0:
            print(f"[NN seed={seed}] epoch {ep+1}/{cfg.nn_epochs}; loss={np.mean(losses):.4f}")

    torch.save(model.state_dict(), model_dir / f"FPProcessNN_seed{seed}.pt")


def read_feature_list(path):
    df = read_table(path)
    if "feature" not in df.columns:
        raise ValueError(f"Feature-list file must contain a feature column: {path}")
    features = list(dict.fromkeys(df["feature"].dropna().astype(str).str.strip().tolist()))
    leak = [f for f in features if f in FORBIDDEN_FEATURES]
    if leak:
        raise ValueError(f"Feature list contains leakage columns: {leak}")
    return features


def train_catboost(feature_df, cfg, model_dir, data_dir):
    rows = []
    for target in TARGET_NAMES:
        features = read_feature_list(FEATURE_LIST_FILES[target])
        missing = [c for c in features if c not in feature_df.columns]
        if missing:
            raise ValueError(f"Missing CatBoost features for {target}: {missing[:30]}")
        X = feature_df[features].copy()
        for c in features:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        y = feature_df[target].astype(float)

        for i, seed in enumerate(cfg.seeds):
            print(f"Training CatBoost {target}, seed={seed}")
            model = CatBoostRegressor(
                loss_function="RMSE", eval_metric="RMSE",
                iterations=cfg.cb_iterations, learning_rate=cfg.cb_learning_rate,
                depth=cfg.cb_depth, l2_leaf_reg=cfg.cb_l2_leaf_reg,
                random_strength=cfg.cb_random_strength,
                bootstrap_type="Bernoulli", subsample=cfg.cb_subsample,
                grow_policy="SymmetricTree", random_seed=seed + 1000*i,
                verbose=False, allow_writing_files=False,
            )
            model.fit(X, y, verbose=False)
            model.save_model(str(model_dir / f"catboost_{target}_seed{seed}.cbm"))
        rows.append({"target": target, "n_features": len(features), "features": ";".join(features)})
    pd.DataFrame(rows).to_csv(data_dir / "catboost_training_summary.csv", index=False, encoding="utf-8-sig")


def resolve_training_file(cfg):
    p = Path(cfg.expanded_training_file) if cfg.expanded_training_file else Path(cfg.run_dir)/"data"/f"expanded_training_data_cycle_{cfg.cycle:02d}.xlsx"
    if not p.exists():
        raise FileNotFoundError(f"Expanded training file not found: {p}")
    return p



def ensure_shift_and_success_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make data compatible with original data_process() and success classifier.

    data_process() requires 'stoke_shift'. For analysis we also keep
    'Stokes_shift_nm'. If success is absent, it is left as NaN here and can
    be filled upstream in script 01.
    """
    out = df.copy()

    if "stoke_shift" not in out.columns:
        if "Stokes_shift_nm" in out.columns:
            out["stoke_shift"] = out["Stokes_shift_nm"]
        elif "Em_max_nm" in out.columns and "Ex_max_nm" in out.columns:
            out["stoke_shift"] = out["Em_max_nm"] - out["Ex_max_nm"]

    if "Stokes_shift_nm" not in out.columns:
        if "stoke_shift" in out.columns:
            out["Stokes_shift_nm"] = out["stoke_shift"]
        elif "Em_max_nm" in out.columns and "Ex_max_nm" in out.columns:
            out["Stokes_shift_nm"] = out["Em_max_nm"] - out["Ex_max_nm"]

    if "success" in out.columns:
        out["success"] = pd.to_numeric(out["success"], errors="coerce")

    return out


def infer_success_feature_columns(feature_df: pd.DataFrame) -> list:
    """
    Infer numeric CatBoost features for success classification.

    We deliberately exclude targets, target-derived columns, meta columns,
    recipe identifiers, and the success label itself.
    """
    exclude = set(FORBIDDEN_FEATURES) | {
        "success",
        "candidate_id",
        "experiment_id",
        "cycle",
        "target_color",
        "selection_type",
        "data_source",
        "recipe_key",
        "precursor_1", "precursor_2", "reagent_1",
        "precursor_1_key", "precursor_2_key", "reagent_1_key",
        "p1_compound_name", "p1_compound_key", "p1_SMILES",
        "p2_compound_name", "p2_compound_key", "p2_SMILES",
        "compound_name", "compound_key", "SMILES",
        "notes",
    }

    cols = []
    for c in feature_df.columns:
        if c in exclude:
            continue
        if c.startswith("pred_"):
            continue
        if c.startswith("observed_"):
            continue

        if pd.api.types.is_numeric_dtype(feature_df[c]):
            if feature_df[c].notna().any():
                cols.append(c)

    if len(cols) == 0:
        raise ValueError("No numeric features inferred for success classifier.")

    return cols



def read_external_success_feature_list(cfg: Cfg):
    """
    Read externally selected success-classifier feature list.

    Expected CSV/XLSX format:
        feature

    Returns None if disabled or file does not exist.
    """
    if not getattr(cfg, "use_external_success_feature_list", False):
        return None

    path = Path(getattr(cfg, "external_success_feature_list_path", ""))
    if not path.exists():
        print(f"[INFO] External success feature list not found: {path}. Falling back to inferred full features.")
        return None

    df = read_table(str(path))
    if "feature" not in df.columns:
        raise ValueError(f"External success feature list must contain column 'feature': {path}")

    features = df["feature"].dropna().astype(str).str.strip().tolist()
    features = [f for f in features if f]
    features = list(dict.fromkeys(features))

    if not features:
        raise ValueError(f"External success feature list is empty: {path}")

    print(f"[INFO] Loaded external selected success features: {len(features)} from {path}")
    return features


def train_success_classifier(feature_df_all: pd.DataFrame, cfg: Cfg, model_dir: Path, data_dir: Path):
    """
    Train CatBoostClassifier ensemble for P_success.

    If an external selected feature list is provided, use it.
    Otherwise, infer numeric non-leakage features from the full feature table.
    """
    if "success" not in feature_df_all.columns:
        print("[WARNING] No success column found. Success classifier will not be trained.")
        return []

    df = feature_df_all.dropna(subset=["success"]).copy()
    if len(df) == 0:
        print("[WARNING] No labeled success rows. Success classifier will not be trained.")
        return []

    y = (pd.to_numeric(df["success"], errors="coerce") >= cfg.success_positive_threshold).astype(int)

    if y.nunique() < 2:
        raise ValueError(
            f"Success classifier requires both positive and negative samples, but got labels: {y.value_counts().to_dict()}"
        )

    external_features = read_external_success_feature_list(cfg)

    if external_features is not None:
        missing = [f for f in external_features if f not in df.columns]
        if missing:
            raise ValueError(
                "External selected success feature list contains features not found "
                f"in current feature table. Examples: {missing[:30]}"
            )
        features = external_features
        feature_source = "external_selected_feature_list"
    else:
        features = infer_success_feature_columns(df)
        feature_source = "inferred_full_numeric_features"

    X = df[features].copy()
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

    print(f"[INFO] Success classifier feature source: {feature_source}")
    print(f"[INFO] Success classifier n_features: {len(features)}")
    print(f"[INFO] Success classifier label counts: {y.value_counts().to_dict()}")

    models = []
    for si, seed in enumerate(cfg.seeds):
        print(f"Training CatBoost success classifier: seed={seed}")
        clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=cfg.cb_iterations,
            learning_rate=cfg.cb_learning_rate,
            depth=cfg.cb_depth,
            l2_leaf_reg=cfg.cb_l2_leaf_reg,
            random_strength=cfg.cb_random_strength,
            bootstrap_type="Bernoulli",
            subsample=cfg.cb_subsample,
            random_seed=seed + 2000 * si,
            verbose=False,
            allow_writing_files=False,
        )
        clf.fit(X, y, verbose=False)
        clf.save_model(str(model_dir / f"catboost_success_seed{seed}.cbm"))
        models.append(clf)

    # This is the file read by 03_ax_bo_search_cycle_success.py.
    feature_file = data_dir / cfg.success_feature_file_name
    pd.DataFrame({"feature": features}).to_csv(feature_file, index=False, encoding="utf-8-sig")

    # Traceability copy.
    selected_copy = data_dir / "success_classifier_feature_list_used_for_training.csv"
    pd.DataFrame({"feature": features}).to_csv(selected_copy, index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "task": "success_classifier",
        "feature_source": feature_source,
        "n_rows": len(df),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "n_features": len(features),
        "feature_file": str(feature_file),
        "feature_file_copy": str(selected_copy),
        "model_pattern": "catboost_success_seed*.cbm",
    }])
    summary.to_csv(data_dir / "success_classifier_training_summary.csv", index=False, encoding="utf-8-sig")

    return models



def main():
    cfg = Cfg()
    set_seed(cfg.random_seed)
    run_dir = Path(cfg.run_dir)
    model_dir, data_dir = run_dir/"models", run_dir/"data"
    model_dir.mkdir(parents=True, exist_ok=True); data_dir.mkdir(parents=True, exist_ok=True)

    train_file = resolve_training_file(cfg)
    print(f"Loading expanded training data: {train_file}")

    raw_all = read_table(str(train_file))
    raw_all = ensure_shift_and_success_columns(raw_all)

    print("\n[CHECK] Expanded training data columns:")
    print(list(raw_all.columns))
    print(f"[CHECK] Expanded training data rows: {len(raw_all)}")
    if "success" in raw_all.columns:
        print("[CHECK] success label counts:")
        print(raw_all["success"].value_counts(dropna=False))
    else:
        print("[WARNING] success column is absent in expanded training data.")

    # ============================================================
    # A. Train success classifier using all rows with success labels.
    #    This step must not depend on data_process(), because data_process()
    #    is designed for property regression and may remove non-emissive rows
    #    if target values are absent.
    # ============================================================
    success_train_df = raw_all.copy()
    if "success" not in success_train_df.columns:
        print("[WARNING] No success column found in expanded data. Success classifier will not be trained.")
        success_feature_df = None
    else:
        success_train_df = success_train_df.dropna(subset=["success"]).copy()
        recipe_cols = ["precursor_1", "precursor_2", "reagent_1", "temperature", "time_h", "pre_1_mmol", "pre_2_mmol"]
        success_train_df = success_train_df.dropna(subset=recipe_cols).copy()

        print(f"\nSuccess-classifier training rows: {len(success_train_df)}")
        print(success_train_df["success"].value_counts(dropna=False))

        if len(success_train_df) > 0:
            print("\nGenerating CatBoost feature table for success classifier...")
            smiles_compound = pd.read_excel(cfg.smiles_path, sheet_name="compound")
            compound_feat_df, fp_objs = build_compound_feature_table(smiles_compound)
            success_feature_df = build_dataset_features(success_train_df, compound_feat_df, fp_objs, has_targets=True)

            success_feat_path = data_dir/f"catboost_training_features_success_cycle_{cfg.cycle:02d}.csv"
            success_feature_df.to_csv(success_feat_path, index=False, encoding="utf-8-sig")

            print("\nTraining P_success classifier...")
            train_success_classifier(success_feature_df, cfg, model_dir, data_dir)
        else:
            success_feature_df = None
            print("[WARNING] No usable success-labeled rows after cleaning. Success classifier will not be trained.")

    # ============================================================
    # B. Train property regressors.
    #    Property regression still uses data_process(), because Ex/Em/QY are required.
    # ============================================================
    raw_for_property = ensure_shift_and_success_columns(raw_all)
    train_df = data_process(raw_for_property).reset_index(drop=True)
    train_df = ensure_shift_and_success_columns(train_df)

    required = ["precursor_1","precursor_2","reagent_1","temperature","time_h","pre_1_mmol","pre_2_mmol","stoke_shift"] + TARGET_NAMES
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise ValueError(f"Property training data missing columns: {missing}")

    if cfg.property_train_success_only and "success" in train_df.columns:
        prop_mask = pd.to_numeric(train_df["success"], errors="coerce") >= cfg.success_positive_threshold
        property_df = train_df.loc[prop_mask].reset_index(drop=True)
        if len(property_df) < 20:
            print(f"[WARNING] Only {len(property_df)} success=1 rows. Falling back to all rows for property regression.")
            property_df = train_df.copy().reset_index(drop=True)
    else:
        property_df = train_df.copy().reset_index(drop=True)

    print(f"\nProperty-regression training rows: {len(property_df)} / property-measured rows: {len(train_df)} / all expanded rows: {len(raw_all)}")
    if "success" in train_df.columns:
        print("[CHECK] property measured rows success counts:")
        print(train_df["success"].value_counts(dropna=False))
        print("[CHECK] property-regression rows success counts:")
        print(property_df["success"].value_counts(dropna=False))

    y_scaler = StandardScaler().fit(property_df[TARGET_NAMES].astype(float).values)
    with open(model_dir/"y_scaler.pkl", "wb") as f:
        pickle.dump(y_scaler, f)

    smiles = pd.read_excel(cfg.smiles_path)
    featurizer = CarbonDotFeaturizer(smiles, get_solvent_polarity_data())
    dataset = FPProcessDataset(property_df, featurizer, has_y=True)

    print("\nTraining FPProcessNN ensemble...")
    for seed in cfg.seeds:
        train_one_nn(seed, dataset, y_scaler, cfg, model_dir)

    print("\nGenerating CatBoost feature table for property-regression rows...")
    # Reuse feature objects if not already made above.
    if "compound_feat_df" not in locals() or "fp_objs" not in locals():
        smiles_compound = pd.read_excel(cfg.smiles_path, sheet_name="compound")
        compound_feat_df, fp_objs = build_compound_feature_table(smiles_compound)

    feature_df_prop = build_dataset_features(property_df, compound_feat_df, fp_objs, has_targets=True)
    feat_prop_path = data_dir/f"catboost_training_features_property_cycle_{cfg.cycle:02d}.csv"
    feature_df_prop.to_csv(feat_prop_path, index=False, encoding="utf-8-sig")

    print("\nTraining CatBoost property-regression ensemble...")
    train_catboost(feature_df_prop, cfg, model_dir, data_dir)

    training_summary = pd.DataFrame([{
        "cycle": cfg.cycle,
        "n_all_expanded_rows": len(raw_all),
        "n_success_classifier_rows": int(len(success_train_df)) if "success_train_df" in locals() else 0,
        "n_property_measured_rows": len(train_df),
        "n_property_training_rows": len(property_df),
        "property_train_success_only": cfg.property_train_success_only,
        "n_success_all": int((pd.to_numeric(raw_all.get("success", pd.Series(dtype=float)), errors="coerce") >= cfg.success_positive_threshold).sum()) if "success" in raw_all.columns else None,
        "n_failure_all": int((pd.to_numeric(raw_all.get("success", pd.Series(dtype=float)), errors="coerce") < cfg.success_positive_threshold).sum()) if "success" in raw_all.columns else None,
        "model_dir": str(model_dir),
        "feature_property_file": str(feat_prop_path),
    }])
    training_summary.to_csv(data_dir/"training_summary.csv", index=False, encoding="utf-8-sig")

    with open(data_dir/"train_config.json", "w", encoding="utf-8") as f:
        json.dump({**asdict(cfg), "training_file": str(train_file), "model_dir": str(model_dir)}, f, ensure_ascii=False, indent=2)

    print(f"\nTraining complete. Models saved to: {model_dir}")


if __name__ == "__main__":
    main()
