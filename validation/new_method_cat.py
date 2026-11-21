import warnings
from pathlib import Path
import re
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski

from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from catboost import CatBoostRegressor

# =========================================================
# paths
# =========================================================
SMILES_FILE = "./data/compound_smiles.xlsx"
DATA_FILE = "./data/output_with_molar_amount_pubchem.xlsx"

OUT_DIR = Path("./cd_cv_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_CSV = OUT_DIR / "cd_hydrothermal_features.csv"
FEATURE_XLSX = OUT_DIR / "cd_hydrothermal_features.xlsx"

CV_LONG_CSV = OUT_DIR / "catboost_repeat3x5_long.csv"
CV_SUMMARY_CSV = OUT_DIR / "catboost_repeat3x5_summary.csv"
OOF_PRED_CSV = OUT_DIR / "catboost_repeat3x5_oof_predictions.csv"

N_BITS = 256
RADIUS = 2
SEED = 42

# =========================================================
# helpers
# =========================================================
def norm_name(x: str) -> str:
    return str(x).strip().lower()

def safe_div(a, b, eps=1e-8):
    return a / (b + eps)

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
    "fg_hydroxyl": "[OX2H]",
    "fg_phenol": "c[OX2H]",
    "fg_carboxylic_acid": "[CX3](=O)[OX2H1]",
    "fg_ester": "[CX3](=O)[OX2][#6]",
    "fg_aldehyde": "[CX3H1](=O)[#6]",
    "fg_ketone": "[#6][CX3](=O)[#6]",
    "fg_carbonyl_any": "[CX3]=[OX1]",
    "fg_primary_amine": "[NX3;H2][#6]",
    "fg_secondary_amine": "[NX3;H1]([#6])[#6]",
    "fg_tertiary_amine": "[NX3;H0]([#6])([#6])[#6]",
    "fg_amide": "[NX3][CX3](=[OX1])[#6]",
    "fg_imide_or_urea_like": "[NX3][CX3](=[OX1])[NX3]",
    "fg_nitro": "[$([NX3](=O)=O),$([N+](=O)[O-])]",
    "fg_nitrile": "[CX2]#N",
    "fg_thiol": "[SX2H]",
    "fg_sulfide": "[#16X2][#6]",
    "fg_sulfonic_acid": "[SX4](=O)(=O)[OX2H]",
    "fg_phosphate_like": "P(=O)(O)(O)",
    "fg_aromatic_ring": "a1aaaaa1",
    "fg_heteroaromatic": "[a;r][n,o,s]",
}

def calc_mol_features(mol, prefix: str):
    feats = {}
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

def build_dataset_features(exp_df: pd.DataFrame, compound_feat_df: pd.DataFrame, fp_objs: dict):
    exp = exp_df.copy()
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
        raise ValueError(f"Missing merged SMILES/features: precursor_1 missing={missing1}, precursor_2 missing={missing2}")

    solv_map = solvent_feature_dict()
    solv_df = pd.DataFrame([solv_map[k] for k in out["reagent_1_key"]], index=out.index)
    out = pd.concat([out, solv_df], axis=1)

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

    out["reagent_to_precursor_vol_mass_ratio"] = safe_div(out["reagent_1_mL"], out["mass_sum_g"])
    out["temp_time"] = out["temperature"] * out["time_h"]
    out["temp_over_time"] = safe_div(out["temperature"], out["time_h"])
    out["same_precursor_flag"] = (out["precursor_1_key"] == out["precursor_2_key"]).astype(int)

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

    for col in continuous_base + fg_base:
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
        (((out["p1_fg_primary_amine_has"] + out["p1_fg_secondary_amine_has"] + out["p1_fg_tertiary_amine_has"]) > 0) &
         (out["p2_fg_carboxylic_acid_has"] > 0)).astype(int)
        +
        (((out["p2_fg_primary_amine_has"] + out["p2_fg_secondary_amine_has"] + out["p2_fg_tertiary_amine_has"]) > 0) &
         (out["p1_fg_carboxylic_acid_has"] > 0)).astype(int)
    )

    tanimoto_list, inter_counts, union_counts, xor_counts = [], [], [], []
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

    if "stoke_shift" not in out.columns:
        out["stoke_shift"] = out["Em_max_nm"] - out["Ex_max_nm"]

    return out

def build_feature_matrix(final_df: pd.DataFrame):
    target_cols = ["Ex_max_nm", "Em_max_nm", "QY_%"]
    if "stoke_shift" in final_df.columns:
        target_cols.append("stoke_shift")

    meta_cols = [
        "precursor_1", "precursor_2", "reagent_1",
        "precursor_1_key", "precursor_2_key", "reagent_1_key",
        "p1_compound_name", "p1_compound_key", "p1_SMILES",
        "p2_compound_name", "p2_compound_key", "p2_SMILES",
    ]
    exclude_cols = set(meta_cols + target_cols)

    X = final_df[[c for c in final_df.columns if c not in exclude_cols]].copy()
    y = final_df[target_cols].copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)

    return X, y, target_cols

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names):
    out = {}
    r2s, maes, rmses = [], [], []
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp)
        out[f"{name}_MAE"] = mae
        out[f"{name}_RMSE"] = rmse
        out[f"{name}_R2"] = r2
        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    out["MAE_mean"] = float(np.mean(maes))
    out["RMSE_mean"] = float(np.mean(rmses))
    out["R2_mean"] = float(np.mean(r2s))
    return out

def summarize_mean_std(df_long: pd.DataFrame):
    metric_cols = [c for c in df_long.columns if c not in ["rep", "fold", "n_train", "n_test"]]
    rows = []
    for col in metric_cols:
        vals = pd.to_numeric(df_long[col], errors="coerce").values
        rows.append({
            "metric": col,
            "mean": np.nanmean(vals),
            "std": np.nanstd(vals),
            "mean±std": f"{np.nanmean(vals):.3f} ± {np.nanstd(vals):.3f}"
        })
    return pd.DataFrame(rows)

def run_catboost_repeated_kfold(X: pd.DataFrame, y: pd.DataFrame, target_names):
    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED)

    rows = []
    oof = y.copy()
    for c in target_names:
        oof[f"pred_{c}"] = np.nan

    for run_id, (tr_idx, te_idx) in enumerate(rkf.split(X), 1):
        rep = (run_id - 1) // 5 + 1
        fold = (run_id - 1) % 5 + 1

        X_tr = X.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]

        model = CatBoostRegressor(
            loss_function="MultiRMSE",
            eval_metric="MultiRMSE",
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5.0,
            random_strength=1.0,
            bagging_temperature=1.0,
            subsample=0.9,
            bootstrap_type="Bernoulli",
            grow_policy="SymmetricTree",
            random_seed=SEED + run_id,
            verbose=False,
            allow_writing_files=False,
        )

        model.fit(
            X_tr, y_tr,
            eval_set=(X_te, y_te),
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=False,
        )

        pred_te = model.predict(X_te)
        pred_te = np.asarray(pred_te, dtype=float)
        if pred_te.ndim == 1:
            pred_te = pred_te.reshape(-1, len(target_names))

        for j, c in enumerate(target_names):
            oof.loc[oof.index[te_idx], f"pred_{c}"] = pred_te[:, j]

        met = calc_metrics(y_te.values, pred_te, target_names)
        row = {"rep": rep, "fold": fold, "n_train": len(tr_idx), "n_test": len(te_idx)}
        row.update(met)
        rows.append(row)

        print(
            f"[Rep {rep}/3 | Fold {fold}/5] "
            f"R2_mean={met['R2_mean']:.3f} | "
            + " | ".join([f"{c}_R2={met[f'{c}_R2']:.3f}" for c in target_names])
        )

    df_long = pd.DataFrame(rows)
    df_summary = summarize_mean_std(df_long)
    return df_long, df_summary, oof

def main():
    df_smiles = pd.read_excel(SMILES_FILE, sheet_name="compound")
    df_exp = pd.read_excel(DATA_FILE)

    compound_feat_df, fp_objs = build_compound_feature_table(df_smiles, n_bits=N_BITS)
    final_df = build_dataset_features(df_exp, compound_feat_df, fp_objs)
    X, y, target_names = build_feature_matrix(final_df)

    final_df.to_csv(FEATURE_CSV, index=False, encoding="utf-8-sig")
    final_df.to_excel(FEATURE_XLSX, index=False)

    print("=" * 80)
    print("Feature table ready")
    print(f"Rows: {final_df.shape[0]}")
    print(f"All columns: {final_df.shape[1]}")
    print(f"Feature columns: {X.shape[1]}")
    print(f"Targets: {target_names}")
    print("=" * 80)

    df_long, df_summary, oof = run_catboost_repeated_kfold(X, y, target_names)

    df_long.to_csv(CV_LONG_CSV, index=False, encoding="utf-8-sig")
    df_summary.to_csv(CV_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    oof.to_csv(OOF_PRED_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Repeated 3x5 KFold summary")
    print(df_summary.to_string(index=False))
    print("=" * 80)
    print(f"Saved: {CV_LONG_CSV}")
    print(f"Saved: {CV_SUMMARY_CSV}")
    print(f"Saved: {OOF_PRED_CSV}")
    print(f"Saved: {FEATURE_CSV}")
    print(f"Saved: {FEATURE_XLSX}")

if __name__ == "__main__":
    main()