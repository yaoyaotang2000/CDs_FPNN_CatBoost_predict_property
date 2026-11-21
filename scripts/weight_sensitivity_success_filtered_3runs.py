#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Separately saved FPNN/CatBoost models + Ax multi-objective weight sensitivity.

Workflow
--------
1. Dynamically load the existing 03 Ax search script and its runtime fusion
   predictor. No pre-built integrated-model file is required. The program loads
   separately saved FPProcessNN_seed*.pt, three target-specific
   catboost_<target>_seed*.cbm files, and optional catboost_success_seed*.cbm.
2. Independently run Ax for every alpha/beta combination using the same budget.
3. Merge and deduplicate all recipes visited by all searches into a shared pool.
4. Re-score the shared pool under every weight combination.
5. Report raw objectives, color-window hit, joint-target attainment,
   Pareto coverage, 2-D hypervolume and Top-k recommendation stability.

Important
---------
- alpha is the QY weight; 1-alpha is the Stokes-shift weight.
- beta is the UCB exploration coefficient (called kappa in some plots/texts).
- The separately saved models are fused at prediction time. For every target,
  predictions from all FPNN seeds and the corresponding CatBoost seeds are
  pooled with equal member weight; their mean is used as the property prediction.
- The surrogate models are not retrained for each weight. Weights belong to the
  acquisition function, not to FPNN or CatBoost training.
- By default, the primary reviewer analysis varies alpha while holding the
  manuscript beta=0.2 fixed. Set run_beta_sensitivity=True for a 2-D grid.
"""

from __future__ import annotations

# Allow the documented "python scripts/..." entry point from a checkout.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass, asdict
from pathlib import Path
import importlib.util
import json
import math
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class SensitivityCfg:
    # Exact filenames supplied by the user. Both files are required because the
    # 03 script calls feature-engineering functions defined in 02 but does not
    # import them itself.
    search_script_path: str = "./validation/03_ax_bo_search_cycle_success.py"
    training_script_path: str = "./validation/02_train_fp_catboost_cycle_success_selected_features.py"

    output_dir: str = "./ax_fpnn_catboost_weight_sensitivity_results_beta"

    # Primary analysis for reviewer Comment 5.
    alpha_grid: tuple = (0.5, 0.7)
    fixed_beta: float = 0.20

    # Optional supplementary alpha-beta analysis.
    run_beta_sensitivity: bool = True
    beta_grid: tuple = (0.0, 0.10, 0.20, 0.5, 1.0, 2.0)

    # Ax budget for each independent seed.  Each seed is one statistical
    # replicate; do not use runs_per_color>1 here because the original 03
    # function does not attach the inner run number to every evaluation row.
    total_trials: int = 100
    # Three genuinely independent Ax repetitions. Across-run mean and sample
    # SD (ddof=1) are calculated from these three seeds.
    replicate_seeds: tuple = (42, 43, 44)

    # Experimental rows marked as unsuccessful must not enter property-metric
    # analyses. This is the measured label, not the predicted P_success value.
    success_column: str = "success"
    successful_label: int = 1
    require_success_column: bool = True

    # Top-k is used only for sensitivity statistics, not experimental batch size.
    top_k: int = 10

    # Analyze all colors even if the online stopping file marks some as stopped.
    # This is recommended for a controlled sensitivity experiment.
    colors: tuple = ("Blue", "Green", "Yellow", "Orange")

    # Corrected orange window.
    color_windows: dict | None = None

    # Application-defined raw-objective thresholds. State these before looking
    # at the sensitivity result; do not tune them to favor alpha=0.7.
    qy_target_percent: float = 20.0
    stokes_target_nm: float = 80.0

    # Recommendation stability is reported relative to the manuscript alpha.
    reference_alpha: float = 0.7
    reference_beta: float = 0.20

    # Whether each alpha-beta Ax search starts from the same RNG seed.
    paired_random_seed: bool = True
    random_seed: int = 42

    # Use diverse precursor-pair selection for final top-k tables.
    enforce_pair_diversity: bool = False

    dpi: int = 400

    # Model-file validation. A single FPNN seed plus a single CatBoost seed is
    # supported, but then the reported std is only their two-model disagreement,
    # not a well-calibrated epistemic uncertainty estimate.
    require_success_model: bool = True
    warn_if_single_seed: bool = True

    def __post_init__(self):
        if self.color_windows is None:
            self.color_windows = {
                "Blue": (440.0, 480.0),
                "Green": (500.0, 540.0),
                "Yellow": (560.0, 580.0),
                "Orange": (580.0, 630.0),
            }


RECIPE_COLS = [
    "precursor_1", "precursor_2", "mass_1_g", "mass_2_g",
    "temperature", "time_h", "reagent_1",
]

PREDICTION_COLS = [
    "Ensemble_pred_Ex_max_nm",
    "Ensemble_pred_Em_max_nm",
    "Ensemble_pred_QY_%",
    "Ensemble_pred_Stokes_shift_nm",
    "Ensemble_std_Ex_max_nm",
    "Ensemble_std_Em_max_nm",
    "Ensemble_std_QY_%",
    "Ensemble_std_Stokes_shift_nm",
    "P_success",
]


def inspect_separate_model_files(search_cfg, sens_cfg: SensitivityCfg):
    """Validate the separately stored model files before starting costly Ax runs."""
    model_dir = Path(search_cfg.run_dir) / "models"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir.resolve()}")

    nn_files = sorted(model_dir.glob("FPProcessNN_seed*.pt"))
    cb_files = {
        target: sorted(model_dir.glob(f"catboost_{target}_seed*.cbm"))
        for target in ("Ex_max_nm", "Em_max_nm", "QY_%")
    }
    success_files = sorted(model_dir.glob("catboost_success_seed*.cbm"))
    scaler_file = model_dir / "y_scaler.pkl"

    missing = []
    if not nn_files:
        missing.append("FPProcessNN_seed*.pt")
    for target, files in cb_files.items():
        if not files:
            missing.append(f"catboost_{target}_seed*.cbm")
    if sens_cfg.require_success_model and not success_files:
        missing.append("catboost_success_seed*.cbm")
    if missing:
        raise FileNotFoundError(
            "Missing separately saved model files in "
            f"{model_dir.resolve()}: {missing}"
        )

    info = {
        "model_dir": str(model_dir.resolve()),
        "n_fpnn_models": len(nn_files),
        "fpnn_files": [p.name for p in nn_files],
        "n_catboost_Ex_models": len(cb_files["Ex_max_nm"]),
        "catboost_Ex_files": [p.name for p in cb_files["Ex_max_nm"]],
        "n_catboost_Em_models": len(cb_files["Em_max_nm"]),
        "catboost_Em_files": [p.name for p in cb_files["Em_max_nm"]],
        "n_catboost_QY_models": len(cb_files["QY_%"]),
        "catboost_QY_files": [p.name for p in cb_files["QY_%"]],
        "n_success_models": len(success_files),
        "success_files": [p.name for p in success_files],
        "y_scaler_exists": scaler_file.exists(),
        "runtime_fusion": "equal weight over all FPNN and target-specific CatBoost members",
    }

    print("\nSeparate-model preflight check:")
    print(f"  model_dir: {model_dir.resolve()}")
    print(f"  FPNN property models: {len(nn_files)}")
    print(
        "  CatBoost property models: "
        f"Ex={len(cb_files['Ex_max_nm'])}, "
        f"Em={len(cb_files['Em_max_nm'])}, "
        f"QY={len(cb_files['QY_%'])}"
    )
    print(f"  CatBoost success models: {len(success_files)}")
    print(f"  y_scaler.pkl: {'found' if scaler_file.exists() else 'not found; original search fallback will be used'}")

    if sens_cfg.warn_if_single_seed and (
        len(nn_files) == 1
        or min(len(v) for v in cb_files.values()) == 1
    ):
        print(
            "[WARNING] Only one seed is available for at least one model family. "
            "The runtime mean remains a valid FPNN-CatBoost fusion prediction, "
            "but the std mainly reflects FPNN-vs-CatBoost disagreement and should "
            "not be described as a calibrated confidence interval."
        )
    return info


def _looks_like_ax_search_script(path: Path) -> bool:
    """Static check avoids importing unrelated training scripts during discovery."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    required_text = (
        "class EnsemblePredictor",
        "def run_bo_color",
        "def ax_parameters",
    )
    return all(token in text for token in required_text)


def _import_source_file(path: Path):
    """Import Python source even when saved as .txt or with a misplaced future import."""
    module_name = "original_ax_search"
    source = path.read_text(encoding="utf-8-sig", errors="strict")
    future_line = "from __future__ import annotations"
    # Some pasted versions contain helper imports before the future statement,
    # which is a Python SyntaxError. Move it to the first executable line in
    # memory without modifying the user's original search file.
    source = "\n".join(
        line for line in source.splitlines() if line.strip() != future_line
    )
    source = future_line + "\n" + source

    spec = importlib.util.spec_from_loader(module_name, loader=None, origin=str(path))
    if spec is None:
        raise ImportError(f"Cannot create import specification for: {path}")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    sys.modules[module_name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def load_search_module(path: str, training_path: str):
    configured = Path(path).resolve()
    candidates = []
    if configured.exists():
        candidates.append(configured)

    # If the configured file is the training script or has been renamed, find
    # the actual Ax search script in the same directory. Both .py and pasted
    # .txt source files are supported.
    search_dir = configured.parent if configured.parent.exists() else Path.cwd()
    current_file = Path(__file__).resolve()
    for pattern in ("*.py", "*.txt"):
        for candidate in sorted(search_dir.glob(pattern)):
            candidate = candidate.resolve()
            if candidate != current_file and candidate not in candidates:
                candidates.append(candidate)

    matches = [p for p in candidates if _looks_like_ax_search_script(p)]
    if not matches:
        checked = "\n".join(f"  - {p}" for p in candidates[:30])
        raise FileNotFoundError(
            "No Ax search source containing class EnsemblePredictor and "
            "def run_bo_color was found.\n"
            f"Configured path: {configured}\n"
            f"Files checked:\n{checked}\n\n"
            "The sensitivity program must reference the search script, not "
            "02_train_fp_catboost_cycle.py. Save the previously supplied Ax "
            "search code as 03_ax_bo_search_cycle_success.py or set "
            "SensitivityCfg.search_script_path to its actual path."
        )

    # Prefer an exact configured match; otherwise use the first discovered Ax file.
    selected = configured if configured in matches else matches[0]
    print(f"Using Ax search source: {selected}")
    module = _import_source_file(selected)

    required = ["Cfg", "EnsemblePredictor", "run_bo_color", "set_seed"]
    missing = [x for x in required if not hasattr(module, x)]
    if missing:
        raise AttributeError(
            f"Ax source was detected but required definitions are unavailable: {missing}. "
            f"Loaded file: {selected}"
        )
    # The uploaded 03 source uses these two feature-engineering functions but
    # does not define or import them. Load the exact 02 training source and
    # inject the functions into the 03 module namespace.
    train_configured = Path(training_path).resolve()
    train_candidates = []
    if train_configured.exists():
        train_candidates.append(train_configured)
    train_dir = train_configured.parent if train_configured.parent.exists() else selected.parent
    for pattern in ("02*.py", "02*.txt", "*train_fp_catboost*.py", "*train_fp_catboost*.txt"):
        for candidate in sorted(train_dir.glob(pattern)):
            candidate = candidate.resolve()
            if candidate not in train_candidates:
                train_candidates.append(candidate)

    feature_tokens = ("def build_compound_feature_table", "def build_dataset_features")
    train_matches = []
    for candidate in train_candidates:
        try:
            source_text = candidate.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if all(token in source_text for token in feature_tokens):
            train_matches.append(candidate)
    if not train_matches:
        checked = "\n".join(f"  - {p}" for p in train_candidates[:30])
        raise FileNotFoundError(
            "The 03 search source requires feature-engineering functions from "
            "the 02 training source, but no matching 02 file was found.\n"
            f"Configured 02 path: {train_configured}\nFiles checked:\n{checked}"
        )
    train_selected = train_configured if train_configured in train_matches else train_matches[0]
    print(f"Using 02 feature-engineering source: {train_selected}")

    # Use a separate module name so loading 02 cannot overwrite the imported 03.
    training_module_name = "original_fp_catboost_training"
    training_source = train_selected.read_text(encoding="utf-8-sig", errors="strict")
    future_line = "from __future__ import annotations"
    training_source = "\n".join(
        line for line in training_source.splitlines() if line.strip() != future_line
    )
    training_source = future_line + "\n" + training_source
    train_spec = importlib.util.spec_from_loader(
        training_module_name, loader=None, origin=str(train_selected)
    )
    if train_spec is None:
        raise ImportError(f"Cannot create import specification for 02 source: {train_selected}")
    train_module = importlib.util.module_from_spec(train_spec)
    train_module.__file__ = str(train_selected)
    sys.modules[training_module_name] = train_module
    exec(compile(training_source, str(train_selected), "exec"), train_module.__dict__)

    feature_names = [
        "build_compound_feature_table",
        "build_dataset_features",
    ]
    feature_missing = [name for name in feature_names if not hasattr(train_module, name)]
    if feature_missing:
        raise AttributeError(
            f"02 source is missing required feature functions: {feature_missing}. "
            f"Loaded file: {train_selected}"
        )
    for name in feature_names:
        setattr(module, name, getattr(train_module, name))

    # Keep a traceable record of the exact sources used.
    module._sensitivity_search_source = str(selected)
    module._sensitivity_training_source = str(train_selected)
    print("Injected 02 feature functions into 03 search module successfully.")
    return module


def weight_grid(cfg: SensitivityCfg):
    betas = cfg.beta_grid if cfg.run_beta_sensitivity else (cfg.fixed_beta,)
    return [(float(a), float(b)) for a in cfg.alpha_grid for b in betas]


def recipe_id_frame(df: pd.DataFrame):
    missing = [c for c in RECIPE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Evaluation table is missing recipe columns: {missing}")
    out = df.copy()
    out["recipe_id"] = (
        out["precursor_1"].astype(str) + "|" +
        out["precursor_2"].astype(str) + "|" +
        out["mass_1_g"].astype(float).round(6).astype(str) + "|" +
        out["mass_2_g"].astype(float).round(6).astype(str) + "|" +
        out["temperature"].astype(int).astype(str) + "|" +
        out["time_h"].astype(int).astype(str) + "|" +
        out["reagent_1"].astype(str)
    )
    return out


def valid_evaluations(df: pd.DataFrame):
    if len(df) == 0:
        return df.copy()
    out = df.copy()
    # Some retrospective evaluation tables contain experimentally observed
    # success labels. When present, analyze only successfully synthesized
    # samples. Prospective Ax evaluation tables normally have P_success but no
    # measured success; P_success must never be treated as a ground-truth label.
    if "success" in out.columns:
        success = pd.to_numeric(out["success"], errors="coerce")
        out = out.loc[success.eq(1)].copy()
    if "error" in out.columns:
        out = out[out["error"].isna() | (out["error"].astype(str).str.strip() == "")]
    if "exclusion_reason" in out.columns:
        out = out[
            out["exclusion_reason"].isna()
            | (out["exclusion_reason"].astype(str).str.strip() == "")
        ]
    required = RECIPE_COLS + PREDICTION_COLS
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"Valid Ax records are missing prediction columns: {missing}. "
            "Check that predictor.score() completed successfully."
        )
    out = out.dropna(subset=required).copy()
    for c in PREDICTION_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.dropna(subset=PREDICTION_COLS).copy()


def prepare_successful_training_subset(search, search_cfg, sens_cfg, data_dir):
    """Validate the expanded dataset and save its success==1 analysis subset.

    The sensitivity program loads already fitted FPNN/CatBoost model files, so
    this function does not pretend to retrain those models. It provides an
    auditable successful-only dataset for retrospective property analyses and
    verifies that the dataset used by the workflow has an explicit success
    label. If the saved property models were originally trained with success=0
    rows, they must be retrained with script 02 before running this program.
    """
    source_candidates = [
        Path(search_cfg.run_dir) / "data"
        / f"expanded_training_data_cycle_{search_cfg.cycle:02d}.xlsx",
        Path(search_cfg.run_dir) / "data"
        / f"expanded_training_data_cycle_{search_cfg.cycle:02d}.csv",
    ]
    source = next((p for p in source_candidates if p.exists()), None)
    if source is None:
        raise FileNotFoundError(
            "Expanded training dataset was not found. Checked: "
            + ", ".join(str(p) for p in source_candidates)
        )

    df = search.read_table(str(source))
    col = sens_cfg.success_column
    if col not in df.columns:
        if sens_cfg.require_success_column:
            raise ValueError(
                f"Training dataset must contain the measured '{col}' column: {source}"
            )
        successful = df.copy()
        n_failed = 0
    else:
        labels = pd.to_numeric(df[col], errors="coerce")
        invalid = sorted(df.loc[labels.notna() & ~labels.isin([0, 1]), col].unique())
        if invalid:
            raise ValueError(
                f"Column '{col}' must contain only 0/1 labels; invalid values: {invalid[:10]}"
            )
        successful = df.loc[labels.eq(sens_cfg.successful_label)].copy()
        n_failed = int(labels.eq(0).sum())

    if len(successful) == 0:
        raise ValueError("No success==1 rows remain after filtering the training dataset.")

    output = data_dir / "successful_training_data_used_for_analysis.csv"
    successful.to_csv(output, index=False, encoding="utf-8-sig")
    info = {
        "source_file": str(source.resolve()),
        "success_column": col,
        "successful_label": int(sens_cfg.successful_label),
        "n_rows_original": int(len(df)),
        "n_success_1": int(len(successful)),
        "n_success_0_removed": n_failed,
        "successful_subset_file": str(output.resolve()),
        "saved_models_retrained_here": False,
    }
    print(
        "[Success filter] "
        f"original={info['n_rows_original']}, retained success=1={info['n_success_1']}, "
        f"removed success=0={info['n_success_0_removed']}"
    )
    return info


def run_all_searches(search, predictor, search_cfg, sens_cfg: SensitivityCfg):
    """Run independent Ax searches and retain every visited valid recipe."""
    out_dir = Path(sens_cfg.output_dir)
    raw_dir = out_dir / "raw_ax_evaluations"
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_eval = []
    run_manifest = []
    combos = weight_grid(sens_cfg)
    total_jobs = len(combos) * len(sens_cfg.colors) * len(sens_cfg.replicate_seeds)
    job = 0

    # Override only sensitivity-experiment settings. Model locations, feature
    # lists, success gating and synthesis-space definitions remain from 03.
    search_cfg.total_trials = sens_cfg.total_trials
    # One Ax run per call. Independent repetitions are controlled explicitly
    # by replicate_seeds so run-level results can never be mixed or lost.
    search_cfg.runs_per_color = 1

    for alpha, beta in combos:
        search_cfg.alpha = alpha
        search_cfg.ucb_beta = beta
        search_cfg.objective_mode = "ucb_score"

        for run_index, base_seed in enumerate(sens_cfg.replicate_seeds, start=1):
            for color_index, color in enumerate(sens_cfg.colors):
                job += 1
                # Paired design: the same seed is used for the same color under
                # every alpha/beta setting. This isolates the weight effect.
                seed = int(base_seed)
                if not sens_cfg.paired_random_seed:
                    seed += int(alpha * 1000) + int(beta * 10000) + color_index
                search_cfg.random_seed = seed
                search.set_seed(seed)
                print(
                    f"\n[Sensitivity Ax {job}/{total_jobs}] color={color}, "
                    f"alpha={alpha:.2f}, beta={beta:.2f}, "
                    f"replicate={run_index}/{len(sens_cfg.replicate_seeds)}, seed={seed}"
                )
                started = time.time()
                best_df, eval_df, trial_df = search.run_bo_color(
                    color, predictor, search_cfg
                )
                eval_df = eval_df.copy()
                # Overwrite the missing/inner value from 03 with the explicitly
                # controlled independent replicate identity.
                eval_df["ax_run"] = run_index
                eval_df["ax_seed"] = seed
                eval_df["search_alpha"] = alpha
                eval_df["search_beta"] = beta
                eval_df["search_seed"] = seed
                eval_df["search_color"] = color
                eval_df["evaluation_order"] = np.arange(1, len(eval_df) + 1)
                eval_df = recipe_id_frame(eval_df)
                all_eval.append(eval_df)

                for frame in (best_df, trial_df):
                    if len(frame):
                        frame["ax_run"] = run_index
                        frame["ax_seed"] = seed
                        frame["search_alpha"] = alpha
                        frame["search_beta"] = beta
                        frame["search_color"] = color

                tag = f"alpha{alpha:.2f}_beta{beta:.2f}_{color}_seed{seed}"
                eval_df.to_csv(raw_dir / f"ax_evaluations_{tag}.csv", index=False, encoding="utf-8-sig")
                if len(best_df):
                    best_df.to_csv(raw_dir / f"ax_best_{tag}.csv", index=False, encoding="utf-8-sig")
                if len(trial_df):
                    trial_df.to_csv(raw_dir / f"ax_trials_{tag}.csv", index=False, encoding="utf-8-sig")

                run_manifest.append({
                    "color": color, "alpha": alpha, "beta": beta,
                    "ax_run": run_index, "seed": seed,
                    "n_evaluations": len(eval_df),
                    "n_valid_evaluations": len(valid_evaluations(eval_df)),
                    "elapsed_seconds": time.time() - started,
                })

    all_eval = pd.concat(all_eval, ignore_index=True)
    manifest = pd.DataFrame(run_manifest)
    return all_eval, manifest


def build_shared_pool(all_eval: pd.DataFrame):
    """Union of recipes visited under every weight, with one prediction per recipe."""
    valid = valid_evaluations(all_eval)
    valid = recipe_id_frame(valid)
    counts = valid.groupby("recipe_id").size().rename("n_times_visited").reset_index()
    pool = valid.sort_values("evaluation_order").drop_duplicates("recipe_id", keep="first")
    # Search-specific score columns are intentionally removed; all scores are
    # recalculated from fixed ensemble predictions below.
    remove = [
        "target_color", "color_fit", "QY_score_fixed", "Stokes_score_fixed",
        "score_mu", "score_sigma_approx", "base_objective_before_success",
        "objective", "search_alpha", "search_beta", "search_seed",
        "search_color", "evaluation_order", "exclusion_reason", "error",
    ]
    pool = pool.drop(columns=[c for c in remove if c in pool.columns])
    pool = pool.merge(counts, on="recipe_id", how="left")
    return pool.reset_index(drop=True)


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def color_fit_array(em, lower, upper, softness):
    softness = max(float(softness), 1e-8)
    return sigmoid((np.asarray(em) - lower) / softness) * sigmoid(
        (upper - np.asarray(em)) / softness
    )


def minmax_fixed(x, lo, hi):
    return np.clip((np.asarray(x, dtype=float) - lo) / max(hi - lo, 1e-8), 0.0, 1.0)


def score_shared_pool(pool, color, alpha, beta, search_cfg, sens_cfg):
    """Recalculate the exact acquisition equation used by the original search."""
    lower, upper = sens_cfg.color_windows[color]
    out = pool.copy()
    em = out["Ensemble_pred_Em_max_nm"].to_numpy(float)
    fit = color_fit_array(em, lower, upper, search_cfg.color_softness_nm)
    qy_s = minmax_fixed(
        out["Ensemble_pred_QY_%"], search_cfg.qy_min_percent, search_cfg.qy_max_percent
    )
    st_s = minmax_fixed(
        out["Ensemble_pred_Stokes_shift_nm"],
        search_cfg.stokes_min_nm, search_cfg.stokes_max_nm,
    )
    qy_std = out["Ensemble_std_QY_%"].to_numpy(float) / max(
        search_cfg.qy_max_percent - search_cfg.qy_min_percent, 1e-8
    )
    st_std = out["Ensemble_std_Stokes_shift_nm"].to_numpy(float) / max(
        search_cfg.stokes_max_nm - search_cfg.stokes_min_nm, 1e-8
    )
    score_mu = fit * (alpha * qy_s + (1.0 - alpha) * st_s)
    score_sigma = fit * (alpha * qy_std + (1.0 - alpha) * st_std)
    base = score_mu + beta * score_sigma
    p_success = out["P_success"].to_numpy(float)
    objective = p_success * base if search_cfg.use_success_probability else base
    if search_cfg.use_success_hard_gate:
        objective = np.where(p_success >= search_cfg.success_min_probability, objective, 0.0)

    out["analysis_color"] = color
    out["analysis_alpha"] = alpha
    out["analysis_beta"] = beta
    out["color_fit"] = fit
    out["QY_score_fixed"] = qy_s
    out["Stokes_score_fixed"] = st_s
    out["score_mu"] = score_mu
    out["score_sigma_approx"] = score_sigma
    out["base_objective_before_success"] = base
    out["objective"] = objective
    out["inside_color_window"] = (em >= lower) & (em <= upper)
    out["joint_target"] = (
        out["inside_color_window"]
        & (out["Ensemble_pred_QY_%"] >= sens_cfg.qy_target_percent)
        & (out["Ensemble_pred_Stokes_shift_nm"] >= sens_cfg.stokes_target_nm)
    )
    return out


def nondominated_mask(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    order = np.lexsort((-y, -x))
    mask = np.zeros(len(x), dtype=bool)
    best_y = -np.inf
    for i in order:
        if y[i] > best_y + 1e-12:
            mask[i] = True
            best_y = y[i]
    return mask


def normalize_global(x):
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if hi <= lo else np.clip((x - lo) / (hi - lo), 0, 1)


def hypervolume_2d(x, y):
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = np.clip(pts, 0, 1)
    pts = pts[nondominated_mask(pts[:, 0], pts[:, 1])]
    pts = pts[np.argsort(pts[:, 0])]
    hv, previous_x = 0.0, 0.0
    for x_i, y_i in pts:
        hv += max(0.0, x_i - previous_x) * max(0.0, y_i)
        previous_x = max(previous_x, x_i)
    return float(hv)


def select_top_k(scored, k, enforce_pair_diversity=False):
    valid = scored[scored["objective"] > 0].sort_values("objective", ascending=False)
    if not enforce_pair_diversity:
        return valid.head(k).copy()
    rows, used = [], set()
    for _, row in valid.iterrows():
        pair = tuple(sorted([str(row.precursor_1), str(row.precursor_2)]))
        if pair not in used:
            rows.append(row)
            used.add(pair)
        if len(rows) >= k:
            break
    if len(rows) < k:
        chosen = {r.recipe_id for r in rows}
        for _, row in valid.iterrows():
            if row.recipe_id not in chosen:
                rows.append(row)
                chosen.add(row.recipe_id)
            if len(rows) >= k:
                break
    return pd.DataFrame(rows).reset_index(drop=True)


def jaccard(ids_a, ids_b):
    a, b = set(ids_a), set(ids_b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def analyze_independent_runs(all_eval, search_cfg, sens_cfg):
    """Calculate one result per independent Ax seed, then mean +/- SD.

    These SD values quantify BO run-to-run variability.  They are deliberately
    kept separate from the within-Top-k candidate SD reported by the older
    shared-pool analysis.
    """
    valid = recipe_id_frame(valid_evaluations(all_eval))
    if len(valid) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Fixed normalization across every run is required for comparable HV.
    valid["QY_hv_norm"] = normalize_global(valid["Ensemble_pred_QY_%"])
    valid["Stokes_hv_norm"] = normalize_global(
        valid["Ensemble_pred_Stokes_shift_nm"]
    )

    group_cols = [
        "search_color", "search_alpha", "search_beta", "ax_run", "ax_seed"
    ]
    run_rows, selected_rows = [], []
    for keys, group in valid.groupby(group_cols, sort=True, dropna=False):
        color, alpha, beta, run_index, seed = keys
        # A recipe visited repeatedly in one run contributes only once.
        group = group.sort_values("evaluation_order").drop_duplicates(
            "recipe_id", keep="first"
        )
        scored = score_shared_pool(
            group, color, float(alpha), float(beta), search_cfg, sens_cfg
        )
        top = select_top_k(
            scored, sens_cfg.top_k, sens_cfg.enforce_pair_diversity
        )
        if len(top) == 0:
            continue

        # Pareto set is defined within the candidates actually visited in this
        # independent run and inside the requested emission window.
        feasible = scored[scored["inside_color_window"]].copy()
        if len(feasible):
            mask = nondominated_mask(
                feasible["Ensemble_pred_QY_%"],
                feasible["Ensemble_pred_Stokes_shift_nm"],
            )
            pareto_ids = set(feasible.loc[mask, "recipe_id"])
        else:
            pareto_ids = set()

        top = top.copy()
        top.insert(0, "selection_rank", np.arange(1, len(top) + 1))
        top["run_color"] = color
        top["run_alpha"] = float(alpha)
        top["run_beta"] = float(beta)
        top["run_index"] = int(run_index)
        top["run_seed"] = int(seed)
        selected_rows.append(top)

        ids = top["recipe_id"].tolist()
        run_rows.append({
            "color": color,
            "alpha_qy": float(alpha),
            "weight_stokes": 1.0 - float(alpha),
            "ucb_beta": float(beta),
            "ax_run": int(run_index),
            "seed": int(seed),
            "n_valid_visited": len(group),
            "top_k": len(top),
            "mean_QY_percent": top["Ensemble_pred_QY_%"].mean(),
            "within_topk_sd_QY_percent": top["Ensemble_pred_QY_%"].std(ddof=1),
            "mean_Stokes_nm": top["Ensemble_pred_Stokes_shift_nm"].mean(),
            "within_topk_sd_Stokes_nm": top["Ensemble_pred_Stokes_shift_nm"].std(ddof=1),
            "mean_Em_nm": top["Ensemble_pred_Em_max_nm"].mean(),
            "color_window_hit_rate": top["inside_color_window"].mean(),
            "joint_target_rate": top["joint_target"].mean(),
            "mean_P_success": top["P_success"].mean(),
            "mean_objective": top["objective"].mean(),
            "pareto_coverage_rate": np.mean([x in pareto_ids for x in ids]),
            "hypervolume": hypervolume_2d(
                top["QY_hv_norm"], top["Stokes_hv_norm"]
            ),
            "selected_recipe_ids": ";".join(ids),
        })

    run_metrics = pd.DataFrame(run_rows)
    run_selected = (
        pd.concat(selected_rows, ignore_index=True)
        if selected_rows else pd.DataFrame()
    )
    if len(run_metrics) == 0:
        return run_metrics, pd.DataFrame(), run_selected

    return run_metrics, aggregate_independent_runs(run_metrics), run_selected


def aggregate_independent_runs(run_metrics):
    metrics = [
        "mean_QY_percent", "mean_Stokes_nm", "mean_Em_nm",
        "color_window_hit_rate", "joint_target_rate", "mean_P_success",
        "mean_objective", "pareto_coverage_rate", "hypervolume",
    ]
    grouped = run_metrics.groupby(
        ["color", "alpha_qy", "weight_stokes", "ucb_beta"], as_index=False
    )
    mean_part = grouped[metrics].mean().rename(
        columns={m: f"{m}_mean" for m in metrics}
    )
    sd_part = grouped[metrics].std(ddof=1).rename(
        columns={m: f"{m}_sd" for m in metrics}
    )
    count_part = grouped.size().rename(columns={"size": "n_runs"})
    aggregate = mean_part.merge(
        sd_part, on=["color", "alpha_qy", "weight_stokes", "ucb_beta"]
    ).merge(
        count_part, on=["color", "alpha_qy", "weight_stokes", "ucb_beta"]
    )
    for metric in metrics:
        aggregate[f"{metric}_sem"] = (
            aggregate[f"{metric}_sd"] / np.sqrt(aggregate["n_runs"])
        )
        aggregate[f"{metric}_ci95"] = 1.96 * aggregate[f"{metric}_sem"]
    return aggregate


def make_run_variability_figure(run_metrics, aggregate, sens_cfg):
    """Mean +/- SD plus individual seed points for reviewer reporting."""
    if len(run_metrics) == 0:
        return
    fig_dir = Path(sens_cfg.output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
    metrics = [
        ("hypervolume", "Hypervolume"),
        ("joint_target_rate", "Joint target-attainment rate"),
        ("mean_QY_percent", "Mean predicted QY (%)"),
        ("mean_Stokes_nm", "Mean Stokes shift (nm)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), constrained_layout=True)
    colors = dict(zip(sens_cfg.colors, sns.color_palette("colorblind", len(sens_cfg.colors))))
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for color in sens_cfg.colors:
            raw = run_metrics[
                (run_metrics.color == color)
                & np.isclose(run_metrics.ucb_beta, sens_cfg.fixed_beta)
            ]
            agg = aggregate[
                (aggregate.color == color)
                & np.isclose(aggregate.ucb_beta, sens_cfg.fixed_beta)
            ].sort_values("alpha_qy")
            if len(agg) == 0:
                continue
            # Small deterministic horizontal offsets reveal all seed points.
            offset = (sens_cfg.colors.index(color) - 1.5) * 0.008
            ax.scatter(
                raw.alpha_qy + offset, raw[metric], s=17, alpha=.28,
                color=colors[color], edgecolors="none"
            )
            ax.errorbar(
                agg.alpha_qy, agg[f"{metric}_mean"],
                yerr=agg[f"{metric}_sd"], fmt="-o", lw=1.8,
                ms=5, capsize=3, color=colors[color], label=color
            )
        ax.set_xlabel(r"QY weight, $\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sens_cfg.alpha_grid)
    axes[0, 0].legend(frameon=False, ncol=2)
    fig.suptitle(
        "Independent-seed BO weight sensitivity (points: runs; error bars: mean ± SD)",
        fontsize=13, fontweight="bold"
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            fig_dir / f"independent_run_mean_sd.{ext}",
            dpi=sens_cfg.dpi if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def analyze_shared_pool(pool, search_cfg, sens_cfg):
    all_selected, summary_rows, pareto_rows = [], [], []

    # Common normalization for hypervolume only; acquisition retains the exact
    # fixed scaling in the original search code.
    pool = pool.copy()
    pool["QY_hv_norm"] = normalize_global(pool["Ensemble_pred_QY_%"])
    pool["Stokes_hv_norm"] = normalize_global(pool["Ensemble_pred_Stokes_shift_nm"])

    pareto_ids_by_color = {}
    for color in sens_cfg.colors:
        lo, hi = sens_cfg.color_windows[color]
        color_pool = pool[pool["Ensemble_pred_Em_max_nm"].between(lo, hi, inclusive="both")].copy()
        if len(color_pool) == 0:
            pareto_ids_by_color[color] = set()
            continue
        mask = nondominated_mask(
            color_pool["Ensemble_pred_QY_%"],
            color_pool["Ensemble_pred_Stokes_shift_nm"],
        )
        color_pool = color_pool.loc[mask].copy()
        color_pool["color"] = color
        pareto_rows.append(color_pool)
        pareto_ids_by_color[color] = set(color_pool.recipe_id)

    for alpha, beta in weight_grid(sens_cfg):
        for color in sens_cfg.colors:
            scored = score_shared_pool(pool, color, alpha, beta, search_cfg, sens_cfg)
            top = select_top_k(
                scored, sens_cfg.top_k, sens_cfg.enforce_pair_diversity
            )
            top.insert(0, "selection_rank", np.arange(1, len(top) + 1))
            all_selected.append(top)
            pareto_ids = pareto_ids_by_color[color]
            selected_ids = top.recipe_id.tolist()
            summary_rows.append({
                "color": color,
                "alpha_qy": alpha,
                "weight_stokes": 1.0 - alpha,
                "ucb_beta": beta,
                "top_k": len(top),
                "mean_QY_percent": top["Ensemble_pred_QY_%"].mean(),
                "std_QY_percent": top["Ensemble_pred_QY_%"].std(ddof=1),
                "mean_Stokes_nm": top["Ensemble_pred_Stokes_shift_nm"].mean(),
                "std_Stokes_nm": top["Ensemble_pred_Stokes_shift_nm"].std(ddof=1),
                "mean_Em_nm": top["Ensemble_pred_Em_max_nm"].mean(),
                "color_window_hit_rate": top["inside_color_window"].mean(),
                "joint_target_rate": top["joint_target"].mean(),
                "mean_P_success": top["P_success"].mean(),
                "mean_objective": top["objective"].mean(),
                "pareto_coverage_rate": np.mean([x in pareto_ids for x in selected_ids]),
                "hypervolume": hypervolume_2d(top["QY_hv_norm"], top["Stokes_hv_norm"]),
                "selected_recipe_ids": ";".join(selected_ids),
            })

    selected = pd.concat(all_selected, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    pareto = pd.concat(pareto_rows, ignore_index=True) if pareto_rows else pd.DataFrame()

    stability_rows = []
    for color in sens_cfg.colors:
        ref = summary[
            (summary.color == color)
            & np.isclose(summary.alpha_qy, sens_cfg.reference_alpha)
            & np.isclose(summary.ucb_beta, sens_cfg.reference_beta)
        ]
        if len(ref) == 0:
            continue
        ref_ids = ref.iloc[0].selected_recipe_ids.split(";")
        for _, row in summary[summary.color == color].iterrows():
            stability_rows.append({
                "color": color,
                "reference_alpha": sens_cfg.reference_alpha,
                "reference_beta": sens_cfg.reference_beta,
                "alpha_qy": row.alpha_qy,
                "ucb_beta": row.ucb_beta,
                "jaccard_vs_reference": jaccard(
                    ref_ids, row.selected_recipe_ids.split(";")
                ),
            })
    stability = pd.DataFrame(stability_rows)
    return summary, selected, pareto, stability


def make_figures(pool, summary, selected, pareto, stability, sens_cfg):
    fig_dir = Path(sens_cfg.output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="paper", font_scale=1.05)

    # Primary figure: raw QY-Stokes trade-off as alpha changes.
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    for ax, color in zip(axes.flat, sens_cfg.colors):
        s = summary[
            (summary.color == color)
            & np.isclose(summary.ucb_beta, sens_cfg.fixed_beta)
        ].sort_values("alpha_qy")
        ax.plot(s.mean_Stokes_nm, s.mean_QY_percent, "-o", lw=2)
        for _, r in s.iterrows():
            ax.annotate(
                f"α={r.alpha_qy:.1f}",
                (r.mean_Stokes_nm, r.mean_QY_percent),
                xytext=(4, 4), textcoords="offset points", fontsize=8,
            )
        ax.set_title(color)
        ax.set_xlabel("Mean predicted Stokes shift (nm)")
        ax.set_ylabel("Mean predicted QY (%)")
    fig.suptitle(f"Raw-objective trade-off of Top-{sens_cfg.top_k} recommendations (β={sens_cfg.fixed_beta:.2f})", fontsize=14)
    fig.savefig(fig_dir / "fig1_qy_stokes_weight_tradeoff.png", dpi=sens_cfg.dpi, bbox_inches="tight")
    fig.savefig(fig_dir / "fig1_qy_stokes_weight_tradeoff.svg", bbox_inches="tight")
    plt.close(fig)

    # Pareto plot.
    palette = dict(zip(sens_cfg.alpha_grid, sns.color_palette("viridis", len(sens_cfg.alpha_grid))))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    for ax, color in zip(axes.flat, sens_cfg.colors):
        lo, hi = sens_cfg.color_windows[color]
        p = pool[pool["Ensemble_pred_Em_max_nm"].between(lo, hi, inclusive="both")]
        if len(p) > 5000:
            p = p.sample(5000, random_state=42)
        ax.scatter(
            p["Ensemble_pred_Stokes_shift_nm"], p["Ensemble_pred_QY_%"],
            s=8, alpha=.15, color="gray", label="Shared Ax pool",
        )
        if len(pareto):
            pf = pareto[pareto.color == color].sort_values("Ensemble_pred_Stokes_shift_nm")
            ax.plot(
                pf["Ensemble_pred_Stokes_shift_nm"], pf["Ensemble_pred_QY_%"],
                color="black", lw=1.5, label="Predicted Pareto front",
            )
        ss = selected[
            (selected.analysis_color == color)
            & np.isclose(selected.analysis_beta, sens_cfg.fixed_beta)
        ]
        for alpha, g in ss.groupby("analysis_alpha"):
            ax.scatter(
                g["Ensemble_pred_Stokes_shift_nm"], g["Ensemble_pred_QY_%"],
                s=26, color=palette[alpha], label=f"α={alpha:.1f}",
            )
        ax.set_title(color)
        ax.set_xlabel("Predicted Stokes shift (nm)")
        ax.set_ylabel("Predicted QY (%)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle("Shared Ax-visited pool, Pareto front and weight-dependent recommendations", fontsize=14)
    fig.savefig(fig_dir / "fig2_pareto_weight_recommendations.png", dpi=sens_cfg.dpi, bbox_inches="tight")
    fig.savefig(fig_dir / "fig2_pareto_weight_recommendations.svg", bbox_inches="tight")
    plt.close(fig)

    # Raw evaluation metrics vs alpha.
    metrics = [
        ("color_window_hit_rate", "Color-window hit rate"),
        ("joint_target_rate", "Joint-target attainment rate"),
        ("hypervolume", "Normalized 2-D hypervolume"),
        ("pareto_coverage_rate", "Pareto coverage rate"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    fixed = summary[np.isclose(summary.ucb_beta, sens_cfg.fixed_beta)]
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for color, g in fixed.groupby("color"):
            g = g.sort_values("alpha_qy")
            ax.plot(g.alpha_qy, g[metric], "-o", label=color)
        ax.set_xlabel(r"QY weight $\alpha$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sens_cfg.alpha_grid)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"Weight robustness evaluated using unaggregated objectives (β={sens_cfg.fixed_beta:.2f})", fontsize=14)
    fig.savefig(fig_dir / "fig3_raw_metric_sensitivity.png", dpi=sens_cfg.dpi, bbox_inches="tight")
    fig.savefig(fig_dir / "fig3_raw_metric_sensitivity.svg", bbox_inches="tight")
    plt.close(fig)

    # Recommendation overlap relative to manuscript alpha=0.7.
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), constrained_layout=True)
    for ax, color in zip(axes.flat, sens_cfg.colors):
        s = stability[stability.color == color]
        if sens_cfg.run_beta_sensitivity:
            p = s.pivot(index="alpha_qy", columns="ucb_beta", values="jaccard_vs_reference")
            sns.heatmap(p, annot=True, fmt=".2f", cmap="magma", vmin=0, vmax=1,
                        linewidths=.5, ax=ax, cbar=ax is axes.flat[-1])
            ax.set_xlabel(r"UCB $\beta$"); ax.set_ylabel(r"QY weight $\alpha$")
        else:
            s = s.sort_values("alpha_qy")
            ax.plot(s.alpha_qy, s.jaccard_vs_reference, "-o", lw=2)
            ax.set_ylim(-.02, 1.02); ax.set_xticks(sens_cfg.alpha_grid)
            ax.set_xlabel(r"QY weight $\alpha$"); ax.set_ylabel("Top-k Jaccard overlap")
        ax.set_title(color)
    fig.suptitle("Recommendation stability relative to α=0.7, β=0.2", fontsize=14)
    fig.savefig(fig_dir / "fig4_recommendation_stability.png", dpi=sens_cfg.dpi, bbox_inches="tight")
    fig.savefig(fig_dir / "fig4_recommendation_stability.svg", bbox_inches="tight")
    plt.close(fig)


def fresh_search(run_dir, cycle, sens_cfg):
    out_dir = Path(sens_cfg.output_dir)
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    search = load_search_module(
        sens_cfg.search_script_path,
        sens_cfg.training_script_path,
    )

    # Make the sensitivity experiment and all plots use exactly the declared
    # windows, including Orange=580-630 nm.
    search.COLOR_WINDOWS = dict(sens_cfg.color_windows)

    search_cfg = search.Cfg()
    search_cfg.cycle = cycle
    search_cfg.run_dir = str(run_dir)
    search_cfg.observed_history_file = str(run_dir / "data/prior_history.csv")
    search_cfg.random_seed = sens_cfg.random_seed
    search.set_seed(sens_cfg.random_seed)

    model_file_info = inspect_separate_model_files(search_cfg, sens_cfg)

    # Validate the measured success label before any BO sensitivity run and
    # create the successful-only dataset used for retrospective analyses.
    success_filter_info = prepare_successful_training_subset(
        search, search_cfg, sens_cfg, data_dir
    )

    print("Loading separately saved FPNN/CatBoost models for runtime fusion...")
    predictor = search.EnsemblePredictor(search_cfg)

    with open(data_dir / "sensitivity_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "sensitivity": asdict(sens_cfg),
                "original_search": asdict(search_cfg),
                "separate_model_files": model_file_info,
                "success_filter": success_filter_info,
            },
            f, ensure_ascii=False, indent=2,
        )

    all_eval, manifest = run_all_searches(
        search, predictor, search_cfg, sens_cfg
    )
    all_eval.to_csv(data_dir / "all_weight_ax_evaluations.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(data_dir / "ax_weight_run_manifest.csv", index=False, encoding="utf-8-sig")

    # Primary inferential output: one row per independent seed, followed by
    # across-seed mean, SD, SEM and approximate 95% CI.
    run_metrics, run_aggregate, run_selected = analyze_independent_runs(
        all_eval, search_cfg, sens_cfg
    )
    run_metrics.to_csv(
        data_dir / "independent_run_metrics.csv", index=False, encoding="utf-8-sig"
    )
    run_aggregate.to_csv(
        data_dir / "weight_sensitivity_mean_sd_by_run.csv",
        index=False, encoding="utf-8-sig"
    )
    run_selected.to_csv(
        data_dir / "top_candidates_by_independent_run.csv",
        index=False, encoding="utf-8-sig"
    )

    shared_pool = build_shared_pool(all_eval)
    shared_pool.to_csv(data_dir / "shared_ax_visited_prediction_pool.csv", index=False, encoding="utf-8-sig")

    summary, selected, pareto, stability = analyze_shared_pool(
        shared_pool, search_cfg, sens_cfg
    )
    summary.to_csv(data_dir / "weight_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(data_dir / "top_candidates_by_weight.csv", index=False, encoding="utf-8-sig")
    selected.to_excel(data_dir / "top_candidates_by_weight.xlsx", index=False)
    pareto.to_csv(data_dir / "pareto_candidates_by_color.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(data_dir / "recommendation_stability.csv", index=False, encoding="utf-8-sig")

    make_figures(shared_pool, summary, selected, pareto, stability, sens_cfg)
    make_run_variability_figure(run_metrics, run_aggregate, sens_cfg)

    print("\nWeight-sensitivity analysis complete.")
    print(f"All Ax evaluations: {len(all_eval):,}")
    print(f"Unique valid recipes in shared pool: {len(shared_pool):,}")
    print(f"Independent run summaries: {len(run_metrics):,}")
    print(f"Results saved in: {out_dir.resolve()}")
    print("\nSensitivity summary:")
    cols = [
        "color", "alpha_qy", "ucb_beta", "mean_QY_percent",
        "mean_Stokes_nm", "color_window_hit_rate", "joint_target_rate",
        "hypervolume", "pareto_coverage_rate",
    ]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    import argparse
    from workflow.stages import resolve_run

    parser = argparse.ArgumentParser(description="Rescore the published candidate pools, or search a locally fitted cycle.")
    parser.add_argument("--run-dir", type=Path, help="Prepared and trained local cycle for a fresh search.")
    parser.add_argument("--study", choices=("alpha", "beta"), default="beta", help="Grid to use for a fresh search.")
    args = parser.parse_args()
    if args.run_dir:
        run_dir, cycle = resolve_run(args.run_dir)
        import json
        archive = "ax_fpnn_catboost_weight_sensitivity_results" + ("_beta" if args.study == "beta" else "")
        with (Path("results") / archive / "data/settings.json").open() as settings_file:
            settings = json.load(settings_file)["sensitivity"]
        cfg = SensitivityCfg(**settings)
        cfg.search_script_path = SensitivityCfg.search_script_path
        cfg.training_script_path = SensitivityCfg.training_script_path
        cfg.output_dir = str(Path("outputs") / run_dir.name / archive)
        fresh_search(run_dir, cycle, cfg)
    else:
        from analysis.sensitivity import main
        main()
