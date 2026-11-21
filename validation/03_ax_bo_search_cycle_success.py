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
from pathlib import Path
import importlib.util

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
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

from model.model import CarbonDotFeaturizer, data_process

import importlib.util

TARGET_NAMES = ["Ex_max_nm", "Em_max_nm", "QY_%"]
# Morgan fingerprint
N_BITS = 256
RADIUS = 2

@dataclass
class Cfg:
    cycle: int = 18
    initial_training_file: str = "./data/output_with_molar_amount_pubchem.xlsx"
    observed_history_file: str = "./data/observed_history.xlsx"
    base_run_dir: str = "./closed_loop_runs"
    reagent_1_mL_default: float = 15.0
    require_measured_targets: bool = True
    drop_duplicate_recipes: bool = True

    # success label:
    # success=1 means emissive/effective fluorescent CD was obtained;
    # success=0 means weak/non-emissive or invalid fluorescence response.
    success_qy_threshold: float = 3.0
    fill_missing_success_from_qy: bool = True
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
    cycle: int = 18
    run_dir: str = "./closed_loop_runs/cycle_18_20261015_190319"
    smiles_path: str = "./data/compound_smiles.xlsx"
    observed_history_file: str = "./data/observed_history.xlsx"

    precursor_mass: tuple = (0.2, 0.4, 0.6, 0.8, 1.0)
    temperature_values: tuple = (160, 180, 200, 220)
    time_values: tuple = (4, 6, 8)
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
    random.seed(seed);
    np.random.seed(seed)
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)


def read_table(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_excel(p) if p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(p)


def minmax_fixed(x, lo, hi):
    return 0.0 if hi <= lo else float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def color_fit_scalar(em, lower, upper, softness):
    center, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
    outside = max(abs(float(em) - center) - half, 0.0)
    return float(np.exp(-(outside ** 2) / (2.0 * softness ** 2)))


def color_fit_array(em_arr, lower, upper, softness):
    em = np.asarray(em_arr, dtype=float)
    center, half = 0.5 * (lower + upper), 0.5 * (upper - lower)
    outside = np.maximum(np.abs(em - center) - half, 0.0)
    return np.exp(-(outside ** 2) / (2.0 * softness ** 2))


def mass_to_mmol(mass_g, mw):
    return float(mass_g) / float(mw) * 1000.0


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


class FingerprintEncoderSmall(nn.Module):
    def __init__(self, input_dim=679, latent_dim=32, dropout=0.30):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, latent_dim),
                                 nn.ReLU())

    def forward(self, x): return self.net(x.float())


class ProcessEncoderSmall(nn.Module):
    def __init__(self, input_dim=14, out_dim=16, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x): return self.net(x.float())


class FPProcessNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp_encoder = FingerprintEncoderSmall()
        self.proc_encoder = ProcessEncoderSmall()
        self.fusion_mlp = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.30), nn.Linear(64, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(80, 64), nn.GELU(), nn.Dropout(0.30), nn.Linear(64, 3))

    def forward(self, fps_1, fps_2, global_feat):
        f1, f2 = self.fp_encoder(fps_1), self.fp_encoder(fps_2)
        fusion = self.fusion_mlp(torch.cat([f1 + f2, torch.abs(f1 - f2)], dim=-1))
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

class EnsemblePredictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir)
        self.model_dir = self.run_dir / "models"
        self.data_dir = self.run_dir / "data"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        scaler_path = self.model_dir / "y_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.y_scaler = pickle.load(f)
        else:
            train_path = self.data_dir / f"expanded_training_data_cycle_{cfg.cycle:02d}.xlsx"
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
                m = CatBoostRegressor();
                m.load_model(str(p));
                models.append(m)
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
            return self.cache[k]

        df = self.params_to_df(params)
        p1, p2, reagent = df.loc[0, "precursor_1"], df.loc[0, "precursor_2"], df.loc[0, "reagent_1"]

        fp1 = torch.tensor(self.featurizer.fingerprints(p1), dtype=torch.float32).unsqueeze(0).to(self.device)
        fp2 = torch.tensor(self.featurizer.fingerprints(p2), dtype=torch.float32).unsqueeze(0).to(self.device)
        s_feat = self.featurizer.solvent_feat(reagent).float()
        r_feat = self.featurizer.reaction_feat(df.loc[0, "temperature"], df.loc[0, "time_h"]).float()
        pre1, pre2 = float(df.loc[0, "pre_1_mmol"]), float(df.loc[0, "pre_2_mmol"])
        ratio = pre1 / (pre1 + pre2) if (pre1 + pre2) > 0 else 0.5
        global_feat = torch.cat([s_feat, r_feat, torch.tensor([ratio], dtype=torch.float32)]).unsqueeze(0).to(
            self.device)

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
            "Ensemble_pred_Stokes_shift_nm": float(em - ex),
            "Ensemble_std_Ex_max_nm": float(ex_std),
            "Ensemble_std_Em_max_nm": float(em_std),
            "Ensemble_std_QY_%": float(qy_std),
            "Ensemble_std_Stokes_shift_nm": float(np.sqrt(ex_std ** 2 + em_std ** 2)),
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
        score_mu = fit * (self.cfg.alpha * qy_s + (1 - self.cfg.alpha) * stokes_s)

        qy_std_norm = pred["Ensemble_std_QY_%"] / max(self.cfg.qy_max_percent - self.cfg.qy_min_percent, 1e-8)
        stokes_std_norm = pred["Ensemble_std_Stokes_shift_nm"] / max(self.cfg.stokes_max_nm - self.cfg.stokes_min_nm,
                                                                     1e-8)
        score_sigma = fit * (self.cfg.alpha * qy_std_norm + (1 - self.cfg.alpha) * stokes_std_norm)

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
    if "reagent" in h.columns and "reagent_1" not in h.columns:
        h = h.rename(columns={"reagent": "reagent_1"})
    req = ["cycle", "precursor_1", "precursor_2", "reagent_1", "temperature", "time_h", "Ex_max_nm", "Em_max_nm",
           "QY_%"]
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
        {"name": "precursor_1", "type": "choice", "values": precursors, "value_type": "str"},
        {"name": "precursor_2", "type": "choice", "values": precursors, "value_type": "str"},
        {"name": "mass_1_g", "type": "choice", "values": list(cfg.precursor_mass), "value_type": "float"},
        {"name": "mass_2_g", "type": "choice", "values": list(cfg.precursor_mass), "value_type": "float"},
        {"name": "temperature", "type": "choice", "values": list(cfg.temperature_values), "value_type": "int"},
        {"name": "time_h", "type": "choice", "values": list(cfg.time_values), "value_type": "int"},
        {"name": "reagent_1", "type": "choice", "values": REAGENT_LIST, "value_type": "str"},
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
            err = dict(params);
            err.update({"target_color": color, "objective": 0.0, "error": str(e)})
            evals.append(err)
            return 0.0

    for run in range(1, cfg.runs_per_color + 1):
        print(f"\n[Ax BO] color={color}, run={run}/{cfg.runs_per_color}")
        best_params, values, experiment, model = optimize(
            parameters=params_def, evaluation_function=objective,
            objective_name="objective", total_trials=cfg.total_trials, minimize=False
        )
        best = predictor.score(best_params, color);
        best["ax_run"] = run;
        best["selection_route"] = "best_per_ax_run"
        bests.append(best)
        try:
            tr = experiment.fetch_data().df.copy();
            tr["target_color"] = color;
            tr["ax_run"] = run
            trials.append(tr)
        except Exception:
            pass
    return pd.DataFrame(bests), pd.DataFrame(evals), pd.concat(trials, ignore_index=True) if trials else pd.DataFrame()


def select_diverse(eval_df, cfg):
    if len(eval_df) == 0: return pd.DataFrame()
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
                selected.append(r);
                used_pairs.add(pair)
            if len(selected) >= cfg.batch_select_per_color: break
        if len(selected) < cfg.batch_select_per_color:
            for _, r in sub.iterrows():
                selected.append(r)
                if len(selected) >= cfg.batch_select_per_color: break
        temp = pd.DataFrame(selected).reset_index(drop=True)
        temp.insert(0, "selected_order", np.arange(1, len(temp) + 1))
        out_all.append(temp)
    return pd.concat(out_all, ignore_index=True) if out_all else pd.DataFrame()


def plot_best(eval_df, fig_dir, cfg):
    if len(eval_df) == 0: return
    plt.figure(figsize=(8, 5))
    for color, sub in eval_df.groupby("target_color"):
        sub = sub.reset_index(drop=True)
        sub["best_so_far"] = sub["objective"].cummax()
        plt.plot(np.arange(1, len(sub) + 1), sub["best_so_far"], label=color)
    plt.xlabel("Ax objective evaluations")
    plt.ylabel("Best-so-far objective")
    plt.title("Ax BO live-model search trajectory")
    plt.legend();
    plt.tight_layout()
    plt.savefig(fig_dir / "ax_bo_best_objective_by_trial.png", dpi=cfg.dpi)
    plt.close()


def main():
    cfg = Cfg()
    set_seed(cfg.random_seed)
    run_dir = Path(cfg.run_dir)
    data_dir, fig_dir = run_dir / "data", run_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True);
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "ax_bo_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    history = load_history(cfg) if cfg.use_offline_stopping else None
    stopping, history_scored = evaluate_stopping(history, cfg)
    stopping.to_csv(data_dir / "stopping_status_by_color.csv", index=False, encoding="utf-8-sig")
    if history_scored is not None:
        history_scored.to_csv(data_dir / "observed_history_with_scores.csv", index=False, encoding="utf-8-sig")

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
        best_all.append(b);
        eval_all.append(e)
        if len(t): trial_all.append(t)

    best_all = pd.concat(best_all, ignore_index=True) if best_all else pd.DataFrame()
    eval_all = pd.concat(eval_all, ignore_index=True) if eval_all else pd.DataFrame()
    trial_all = pd.concat(trial_all, ignore_index=True) if trial_all else pd.DataFrame()
    selected = select_diverse(eval_all, cfg)

    best_all.to_csv(data_dir / "ax_bo_best_per_run_by_color.csv", index=False, encoding="utf-8-sig")
    eval_all.to_csv(data_dir / "ax_bo_all_evaluations_by_color.csv", index=False, encoding="utf-8-sig")
    trial_all.to_csv(data_dir / "ax_bo_trials_raw_by_color.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(data_dir / "ax_bo_recommendations_by_color.csv", index=False, encoding="utf-8-sig")
    selected.to_excel(data_dir / "ax_bo_recommendations_by_color.xlsx", index=False)
    plot_best(eval_all, fig_dir, cfg)

    show = ["target_color", "selected_order", "precursor_1", "precursor_2",
            "mass_1_g", "mass_2_g", "pre_1_mmol", "pre_2_mmol", "temperature", "time_h", "reagent_1",
            "Ensemble_pred_Ex_max_nm", "Ensemble_pred_Em_max_nm", "Ensemble_pred_QY_%", "Ensemble_pred_Stokes_shift_nm",
            "color_fit", "P_success", "score_mu", "score_sigma_approx", "base_objective_before_success", "objective",
            "exclusion_reason", ]
    show = [c for c in show if c in selected.columns]
    print("\nFinal selected BO recommendations:")
    print(selected[show].to_string(index=False))
    print(f"\nResults saved in: {run_dir}")


if __name__ == "__main__":
    main()