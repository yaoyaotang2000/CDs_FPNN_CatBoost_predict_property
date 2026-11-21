# 01_update_training_data_from_history.py
# -*- coding: utf-8 -*-
"""
Closed-loop Step 1:
Update training data after each experimental cycle.

- Reads initial training data.
- Reads observed_history.xlsx/csv.
- Standardizes columns.
- Calculates pre_1_mmol/pre_2_mmol from mass when missing.
- Adds Stokes_shift_nm.
- Merges initial data + measured observed-history rows.
- Saves expanded_training_data_cycle_XX.xlsx/csv into a timestamped run folder.

Run:
    python 01_update_training_data_from_history.py
"""

from __future__ import annotations
# Allow the documented "python scripts/..." entry point from a checkout.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd


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
class Cfg:
    cycle: int = 1
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


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return pd.read_excel(p) if p.suffix.lower() in [".xlsx", ".xls"] else pd.read_csv(p)


def write_observed_history_template(path: str):
    p = Path(path)
    if p.exists():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "cycle", "experiment_id", "candidate_id", "target_color", "selection_type",
        "precursor_1", "precursor_2", "reagent_1",
        "mass_1_g", "mass_2_g", "pre_1_mmol", "pre_2_mmol", "reagent_1_mL",
        "temperature", "time_h",
        "pred_Ex_max_nm", "pred_Em_max_nm", "pred_QY_%",
        "pred_Stokes_shift_nm", "pred_score_mu", "pred_score_sigma",
        "expected_improvement",
        "Ex_max_nm", "Em_max_nm", "QY_%", "Stokes_shift_nm",
        "success",
        "notes",
    ]
    pd.DataFrame(columns=cols).to_excel(p, index=False)
    print(f"Created observed-history template: {p}")


def standardize(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "reagent": "reagent_1", "mass_1": "mass_1_g", "mass_2": "mass_2_g",
        "Mass1(g)": "mass_1_g", "Mass2(g)": "mass_2_g",
        # Keep original "stoke_shift" because model.model.data_process() expects it.
        "stokes_shift": "stoke_shift",
    }
    for old, new in rename.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})

    if "reagent_1_mL" not in out.columns:
        out["reagent_1_mL"] = cfg.reagent_1_mL_default
    if "pre_1_mmol" not in out.columns:
        out["pre_1_mmol"] = np.nan
    if "pre_2_mmol" not in out.columns:
        out["pre_2_mmol"] = np.nan

    for i, r in out.iterrows():
        p1, p2 = str(r.get("precursor_1", "")), str(r.get("precursor_2", ""))
        if pd.isna(r.get("pre_1_mmol", np.nan)) and "mass_1_g" in out.columns:
            if p1 in MOLECULAR_WEIGHTS and pd.notna(r.get("mass_1_g", np.nan)):
                out.loc[i, "pre_1_mmol"] = float(r["mass_1_g"]) / MOLECULAR_WEIGHTS[p1] * 1000.0
        if pd.isna(r.get("pre_2_mmol", np.nan)) and "mass_2_g" in out.columns:
            if p2 in MOLECULAR_WEIGHTS and pd.notna(r.get("mass_2_g", np.nan)):
                out.loc[i, "pre_2_mmol"] = float(r["mass_2_g"]) / MOLECULAR_WEIGHTS[p2] * 1000.0

    if "Stokes_shift_nm" not in out.columns and {"Em_max_nm", "Ex_max_nm"}.issubset(out.columns):
        out["Stokes_shift_nm"] = out["Em_max_nm"] - out["Ex_max_nm"]
    return out


def recipe_key(df: pd.DataFrame) -> pd.Series:
    keys = []
    for _, r in df.iterrows():
        a1 = r.get("mass_1_g", r.get("pre_1_mmol", np.nan))
        a2 = r.get("mass_2_g", r.get("pre_2_mmol", np.nan))
        keys.append((
            str(r.get("precursor_1", "")), str(r.get("precursor_2", "")),
            str(r.get("reagent_1", "")),
            int(r.get("temperature", -999)) if pd.notna(r.get("temperature", np.nan)) else -999,
            int(r.get("time_h", -999)) if pd.notna(r.get("time_h", np.nan)) else -999,
            round(float(a1), 6) if pd.notna(a1) else None,
            round(float(a2), 6) if pd.notna(a2) else None,
        ))
    return pd.Series(keys, index=df.index)


def make_run_dir(cfg: Cfg) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(cfg.base_run_dir) / f"cycle_{cfg.cycle:02d}_{stamp}"
    for sub in ["data", "models", "figures"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def main(cfg=None):
    cfg = cfg or Cfg()
    run_dir = make_run_dir(cfg)
    data_dir = run_dir / "data"

    initial = standardize(read_table(cfg.initial_training_file), cfg)
    initial["data_source"] = "initial_training"

    history_path = Path(cfg.observed_history_file)
    history = read_table(str(history_path)) if history_path.exists() else pd.DataFrame()
    if len(history):
        history = history.loc[pd.to_numeric(history["cycle"], errors="raise") < cfg.cycle].copy()
    else:
        history = pd.DataFrame(columns=["cycle","target_color","precursor_1","precursor_2","reagent_1","temperature","time_h","Ex_max_nm","Em_max_nm","QY_%","Stokes_shift_nm","success"])
    history.to_csv(data_dir / "prior_history.csv", index=False)
    history = standardize(history, cfg) if len(history) else history

    if len(history) and cfg.require_measured_targets:
        history = history.dropna(subset=["Ex_max_nm", "Em_max_nm", "QY_%"]).copy()
    if len(history):
        history["data_source"] = "observed_history"

    merged = pd.concat([initial, history], ignore_index=True, sort=False)

    required = [
        "precursor_1", "precursor_2", "reagent_1",
        "temperature", "time_h", "pre_1_mmol", "pre_2_mmol",
        "Ex_max_nm", "Em_max_nm", "Stokes_shift_nm", "QY_%", "success",
    ]
    missing = [c for c in required if c not in merged.columns]
    if missing:
        raise ValueError(f"Expanded training data missing required columns: {missing}")

    merged = merged.dropna(subset=required).copy()
    merged["recipe_key"] = recipe_key(merged)
    before = len(merged)
    if cfg.drop_duplicate_recipes:
        merged = merged.drop_duplicates(subset=["recipe_key"], keep="last").reset_index(drop=True)
    after = len(merged)

    xlsx = data_dir / f"expanded_training_data_cycle_{cfg.cycle:02d}.xlsx"
    csv = data_dir / f"expanded_training_data_cycle_{cfg.cycle:02d}.csv"
    merged.to_excel(xlsx, index=False)
    merged.to_csv(csv, index=False, encoding="utf-8-sig")

    manifest = {
        **asdict(cfg),
        "run_dir": str(run_dir),
        "expanded_training_xlsx": str(xlsx),
        "expanded_training_csv": str(csv),
        "n_rows_before_dedup": before,
        "n_rows_after_dedup": after,
        "success_counts": merged["success"].value_counts(dropna=False).to_dict() if "success" in merged.columns else {},
    }
    with open(data_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Expanded training data saved to:\n{xlsx}\n{csv}")
    print(f"Rows before dedup: {before}; after dedup: {after}")
    print(f"Run directory: {run_dir}")
    return run_dir


if __name__ == "__main__":
    from workflow.stages import stage_cli
    stage_cli(1)
