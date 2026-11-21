from __future__ import annotations
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import argparse, sys, hashlib, json
import numpy as np
import pandas as pd

# ============================================================
# 1. Configuration
# ============================================================
INPUT_FILE = Path("./data/validation.csv")
SHEET_NAME = "Sheet1"
STOKES_SOURCE = "derive"  # derive = round(raw Em - raw Ex)
STOPPING_ELIGIBILITY_COLUMN = "include_in_stopping"

COLOR_WINDOWS = {
    "Blue": (440.0, 480.0),
    "Green": (500.0, 540.0),
    "Yellow": (560.0, 580.0),
    "Orange": (580.0, 630.0),
}

ALPHA, COLOR_SOFTNESS_NM = 0.7, 10.0
QY_BOUNDS, STOKES_BOUNDS = (0.0, 100.0), (40.0, 180.0)
ABS_TOLERANCE, REL_TOLERANCE, SCORE_NOISE_FLOOR = 0.02, 0.03, 0.02
PATIENCE_CYCLES, MIN_CYCLES, MAX_CYCLES = 4, 3, 16

TARGETS = {
    "Ex": ("pred_Ex_max_nm", "Ex_max_nm", "nm"),
    "Em": ("pred_Em_max_nm", "Em_max_nm", "nm"),
    "QY": ("pred_QY_%", "QY_%", "percentage points"),
    "Stokes": ("pred_Stokes_shift_nm", "Stokes_shift_nm", "nm"),
}

RECIPE_COLUMNS = [
    "precursor_1", "precursor_2", "reagent_1", "mass_1_g",
    "mass_2_g", "reagent_1_mL", "temperature", "time_h"
]


# ============================================================
# 2. Rounding and data preparation
# ============================================================
def round_half_up(s: pd.Series, digits=0) -> pd.Series:
    """Excel-style ROUND_HALF_UP; digits=0 integer, digits=2 two decimals."""
    q = Decimal("1") if digits == 0 else Decimal("1").scaleb(-digits)
    s = pd.to_numeric(s, errors="coerce")
    return s.map(lambda x: float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP)) if pd.notna(x) else np.nan)


def prepare_values(df: pd.DataFrame, stokes_source: str) -> pd.DataFrame:
    """Apply final calculation precision rules while preserving raw values."""
    d = df.copy()

    # Preserve raw Ex/Em/QY
    for c in ["Ex_max_nm", "Em_max_nm", "QY_%", "pred_Ex_max_nm", "pred_Em_max_nm", "pred_QY_%"]:
        if c not in d: d[c] = np.nan
        d[c + "_raw"] = d[c]

    # Prediction values used in ALL error calculations
    d["pred_Ex_max_nm"] = round_half_up(d["pred_Ex_max_nm_raw"], 0).astype("Int64")
    d["pred_Em_max_nm"] = round_half_up(d["pred_Em_max_nm_raw"], 0).astype("Int64")
    d["pred_QY_%"] = round_half_up(d["pred_QY_%_raw"], 2)

    # Experimental values
    d["Ex_max_nm"] = pd.to_numeric(d["Ex_max_nm_raw"], errors="coerce")
    d["Em_max_nm"] = round_half_up(d["Em_max_nm_raw"], 0).astype("Int64")
    d["QY_%"] = pd.to_numeric(d["QY_%_raw"], errors="coerce")

    # Stokes: derive from RAW Em - RAW Ex, then round to integer
    for prefix in ("", "pred_"):
        em, ex, st = prefix + "Em_max_nm", prefix + "Ex_max_nm", prefix + "Stokes_shift_nm"
        if st not in d: d[st] = np.nan
        d[st + "_raw"] = pd.to_numeric(d[st], errors="coerce")
        derived = d[em + "_raw"] - d[ex + "_raw"]
        d[st + "_derived_raw"] = derived
        d[st + "_stored_minus_derived"] = d[st + "_raw"] - derived
        d[st + "_inconsistent"] = d[st + "_stored_minus_derived"].abs() > 1e-6
        values = derived if stokes_source == "derive" else d[st + "_raw"]
        d[st] = round_half_up(values, 0).astype("Int64")
    return d


# ============================================================
# 3. Scores / hit rate
# ============================================================
def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d[["Em_max_nm", "Stokes_shift_nm", "QY_%"]].isna().any().any():
        raise ValueError("Measured Em, Stokes and QY must be complete.")
    if not d["QY_%"].between(0, 100).all():
        raise ValueError("QY_% must be in [0,100].")

    lo = d["target_color"].map(lambda c: COLOR_WINDOWS[c][0]).astype(float)
    hi = d["target_color"].map(lambda c: COLOR_WINDOWS[c][1]).astype(float)
    em = d["Em_max_nm"].astype(float)
    distance = np.maximum(np.maximum(lo - em, em - hi), 0)

    d["color_fit_obs"] = np.exp(-distance**2 / (2 * COLOR_SOFTNESS_NM**2))
    d["strict_color_hit"] = em.ge(lo) & em.le(hi)
    d["QY_score_fixed_obs"] = ((d["QY_%"] - QY_BOUNDS[0]) / (QY_BOUNDS[1] - QY_BOUNDS[0])).clip(0, 1)
    d["Stokes_score_fixed_obs"] = ((d["Stokes_shift_nm"].astype(float) - STOKES_BOUNDS[0]) /
                                   (STOKES_BOUNDS[1] - STOKES_BOUNDS[0])).clip(0, 1)
    d["current_measured_score"] = d["color_fit_obs"] * (
        ALPHA * d["QY_score_fixed_obs"] + (1 - ALPHA) * d["Stokes_score_fixed_obs"]
    )

    if STOPPING_ELIGIBILITY_COLUMN not in d:
        d[STOPPING_ELIGIBILITY_COLUMN] = True
    else:
        mapping = {"true": True, "1": True, "1.0": True, "yes": True,
                   "false": False, "0": False, "0.0": False, "no": False}
        v = d[STOPPING_ELIGIBILITY_COLUMN].astype(str).str.strip().str.lower().map(mapping)
        if v.isna().any():
            raise ValueError("include_in_stopping must contain True/False or 1/0.")
        d[STOPPING_ELIGIBILITY_COLUMN] = v
    return d


# ============================================================
# 4. Recipe-level errors
# ============================================================
def add_errors(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.insert(0, "history_row", np.arange(1, len(d) + 1))
    d["formula_id"] = ("C" + d["cycle"].astype("Int64").astype(str).str.zfill(2) + "_" +
                       d["target_color"].astype(str) + "_" +
                       (d.groupby(["cycle", "target_color"]).cumcount() + 1).astype(str))

    available = [c for c in RECIPE_COLUMNS if c in d.columns]
    d["duplicate_recipe"] = d.duplicated(RECIPE_COLUMNS, keep=False) if len(available) == len(RECIPE_COLUMNS) else False

    for target, (pred, exp, _) in TARGETS.items():
        e = d[pred].astype(float) - d[exp].astype(float)
        d[f"{target}_error_pred_minus_exp"] = e
        d[f"{target}_absolute_error"] = e.abs()
        d[f"{target}_squared_error"] = e**2
        d[f"{target}_APE_pct"] = np.where(d[exp].abs() > 1e-12, e.abs() / d[exp].abs() * 100, np.nan)
    return d


# ============================================================
# 5. Statistics
# ============================================================
def safe_r2(y, p):
    if len(y) < 2 or np.sum((y - y.mean())**2) <= 0: return np.nan
    return float(1 - np.sum((y - p)**2) / np.sum((y - y.mean())**2))


def safe_pearson(y, p):
    if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0: return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def group_metrics(g: pd.DataFrame) -> dict:
    out = {"n_recipes": len(g)}
    for target, (pred, exp, _) in TARGETS.items():
        v = g[[pred, exp]].dropna()
        out[f"{target}_n_valid"] = len(v)
        names = ["bias", "MAE", "median_AE", "RMSE", "max_AE", "MAPE_pct", "R2", "Pearson_r"]
        if v.empty:
            out.update({f"{target}_{x}": np.nan for x in names})
            continue

        p, y = v[pred].to_numpy(float), v[exp].to_numpy(float)
        e, ae = p - y, np.abs(p - y)
        nz = np.abs(y) > 1e-12
        out.update({
            f"{target}_bias": float(e.mean()),
            f"{target}_MAE": float(ae.mean()),
            f"{target}_median_AE": float(np.median(ae)),
            f"{target}_RMSE": float(np.sqrt(np.mean(e**2))),
            f"{target}_max_AE": float(ae.max()),
            f"{target}_MAPE_pct": float(np.mean(ae[nz] / np.abs(y[nz])) * 100) if nz.any() else np.nan,
            f"{target}_R2": safe_r2(y, p),
            f"{target}_Pearson_r": safe_pearson(y, p),
        })
    return out


def metrics_by_group(df, cols):
    records, grouper = [], cols[0] if len(cols) == 1 else cols
    for key, g in df.groupby(grouper, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(cols, key)); r.update(group_metrics(g)); records.append(r)
    return pd.DataFrame(records)


def cumulative_metrics(df):
    records = []
    for color, d in df.groupby("target_color", sort=True):
        d = d.sort_values("cycle")
        for cycle in sorted(d["cycle"].unique()):
            g = d[d["cycle"] <= cycle]
            r = {"target_color": color, "up_to_cycle": cycle, "n_cycles_observed": g["cycle"].nunique()}
            r.update(group_metrics(g)); records.append(r)
    return pd.DataFrame(records)


# ============================================================
# 6. Stopping trace
# ============================================================
def stopping_trace(df):
    records = []
    for color, sub in df.groupby("target_color", sort=True):
        best, patience, hit_seen, first_stop, eligible_cycles, previous_cycle = None, 0, False, None, 0, None

        for cycle, group in sub.groupby("cycle", sort=True):
            eligible = group[group[STOPPING_ELIGIBILITY_COLUMN]]
            ok = not eligible.empty
            current = float(eligible["current_measured_score"].max()) if ok else np.nan
            previous_best = best
            tolerance = max(ABS_TOLERANCE, REL_TOLERANCE * (best or 0), SCORE_NOISE_FLOOR)
            improvement, significant = np.nan, False

            if ok:
                eligible_cycles += 1
                hit_seen |= bool(eligible["strict_color_hit"].any())
                if best is None:
                    best = current
                else:
                    best = max(best, current)
                    improvement = best - previous_best
                    significant = Decimal(str(best)) - Decimal(str(previous_best)) > Decimal(str(tolerance))
                    patience = 0 if significant else patience + 1

            stop = bool(ok and hit_seen and
                        (cycle >= MAX_CYCLES or (cycle >= MIN_CYCLES and patience >= PATIENCE_CYCLES)))
            if stop and first_stop is None: first_stop = int(cycle)

            reasons = []
            if not ok: reasons.append("no eligible measurement this cycle")
            elif not hit_seen: reasons.append("no strict color hit yet")
            else:
                if cycle >= MAX_CYCLES: reasons.append("maximum cycle budget reached")
                if cycle >= MIN_CYCLES and patience >= PATIENCE_CYCLES: reasons.append("patience reached")

            records.append({
                "target_color": color, "cycle": int(cycle), "current_measured_score": current,
                "best_so_far_score": best, "tolerance": tolerance, "patience_counter": patience,
                "previous_best_score": previous_best, "improvement": improvement,
                "significant_improvement": significant, "strict_hit_seen": hit_seen,
                "stop_rule_this_prefix": stop, "stop_latched": first_stop is not None,
                "first_stop_cycle": first_stop,
                "after_first_stop": first_stop is not None and cycle > first_stop,
                "continue_recommendation_latched": first_stop is None,
                "budget_reached": cycle >= MAX_CYCLES,
                "n_measurements": len(group), "n_eligible_measurements": len(eligible),
                "n_eligible_cycles": eligible_cycles,
                "cycle_gap_before": 0 if previous_cycle is None else int(cycle - previous_cycle - 1),
                "duplicate_recipe_present": bool(group["duplicate_recipe"].any()),
                "reason": "; ".join(reasons) if reasons else "continue",
            })
            previous_cycle = cycle
    return pd.DataFrame(records)


# ============================================================
# 7. Quality / hit-rate summaries
# ============================================================
def quality_summary(df, n_total, n_success):
    rows = [
        {"check": "input_rows_before_success_filter", "value": n_total},
        {"check": "success_eq_1_rows_retained", "value": n_success},
        {"check": "rows_excluded_by_success_filter", "value": n_total - n_success},
        {"check": "unique_cycles", "value": df["cycle"].nunique()},
        {"check": "unique_target_colors", "value": df["target_color"].nunique()},
        {"check": "rows_with_duplicate_recipe", "value": int(df["duplicate_recipe"].sum()) if set(RECIPE_COLUMNS).issubset(df.columns) else "not evaluated on anonymized endpoints"},
    ]
    for _, (pred, exp, _) in TARGETS.items():
        rows += [{"check": f"missing_{pred}", "value": int(df[pred].isna().sum())},
                 {"check": f"missing_{exp}", "value": int(df[exp].isna().sum())}]
    return pd.DataFrame(rows)


def hit_summary(df):
    x = df.groupby("target_color", sort=True)["strict_color_hit"].agg(["count", "sum"]).reset_index()
    x.columns = ["target_color", "n_experiments", "n_hits"]
    x["hit_rate_pct"] = x["n_hits"] / x["n_experiments"] * 100
    overall = pd.DataFrame([{
        "target_color": "All", "n_experiments": len(df),
        "n_hits": int(df["strict_color_hit"].sum()),
        "hit_rate_pct": df["strict_color_hit"].mean() * 100
    }])
    return pd.concat([x, overall], ignore_index=True)


# ============================================================
# 8. Plotting
# ============================================================
def plot_metrics(per_cycle, color_cycle, cumulative, output_dir):
    import matplotlib.pyplot as plt

    for target, (_, _, unit) in TARGETS.items():
        plt.figure(figsize=(7.2, 4.8))
        plt.plot(per_cycle["cycle"], per_cycle[f"{target}_MAE"], marker="o", label="MAE")
        plt.plot(per_cycle["cycle"], per_cycle[f"{target}_RMSE"], marker="s", label="RMSE")
        plt.xlabel("BO cycle"); plt.ylabel(f"Error ({unit})"); plt.title(f"Per-cycle prediction error: {target}")
        plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
        plt.savefig(output_dir / f"cycle_{target}_MAE_RMSE.png", dpi=300); plt.close()

        plt.figure(figsize=(7.2, 4.8))
        for color, g in color_cycle.groupby("target_color", sort=True):
            plt.plot(g["cycle"], g[f"{target}_MAE"], marker="o", label=color)
        plt.xlabel("BO cycle"); plt.ylabel(f"MAE ({unit})"); plt.title(f"Color-specific iterative error: {target}")
        plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
        plt.savefig(output_dir / f"color_cycle_{target}_MAE.png", dpi=300); plt.close()

        plt.figure(figsize=(7.2, 4.8))
        for color, g in cumulative.groupby("target_color", sort=True):
            plt.plot(g["up_to_cycle"], g[f"{target}_MAE"], marker="o", label=color)
        plt.xlabel("Up to BO cycle"); plt.ylabel(f"Cumulative MAE ({unit})")
        plt.title(f"Color-specific cumulative prediction error: {target}")
        plt.legend(); plt.grid(alpha=.25); plt.tight_layout()
        plt.savefig(output_dir / f"color_cumulative_{target}_MAE.png", dpi=300); plt.close()


# ============================================================
# 9. Excel export
# ============================================================
def autosize(writer, name, df):
    ws = writer.sheets[name]
    ws.freeze_panes(1, 2 if name == "Iteration_scores" else 0)
    if len(df.columns): ws.autofilter(0, 0, len(df), len(df.columns) - 1)
    header = writer.book.add_format({"bold": True, "bg_color": "#17365D", "font_color": "white", "text_wrap": True})
    num = writer.book.add_format({"num_format": "0.000000"})
    integer = writer.book.add_format({"num_format": "0"})
    wrap = writer.book.add_format({"text_wrap": True, "valign": "top"})
    ws.set_row(0, 45)

    for i, c in enumerate(df.columns):
        width = max(len(str(c)), df[c].map(lambda x: len(str(x)) if pd.notna(x) else 0).max() if len(df) else 0) + 2
        fmt = num if pd.api.types.is_float_dtype(df[c]) else integer if pd.api.types.is_integer_dtype(df[c]) else None
        if name == "Readme": width, fmt = (30 if i == 0 else 100), wrap
        ws.set_column(i, i, max(12, min(width, 30)) if name != "Readme" else width, fmt)
        ws.write(0, i, c, header)


def export_results(path, sheets, metadata):
    readme = pd.DataFrame({
        "item": [
            "Success filter", "Prediction precision", "Measured precision", "Stokes",
            "Signed error", "MAE", "RMSE", "Color windows", "Score"
        ],
        "definition": [
            "Only success == 1 rows are included in all calculations; Raw_input retains all original rows.",
            "pred_Ex_max_nm and pred_Em_max_nm: ROUND_HALF_UP to integer; pred_QY_%: ROUND_HALF_UP to 2 decimals BEFORE calculation.",
            "Measured Em: ROUND_HALF_UP to integer; measured Ex and QY retain original precision.",
            "Derived from raw Em - raw Ex, then ROUND_HALF_UP to integer.",
            "prediction - experimental value", "mean absolute prediction error",
            "root mean squared prediction error", json.dumps(COLOR_WINDOWS),
            "color_fit * [0.7*clip(QY/100,0,1) + 0.3*clip((Stokes-40)/140,0,1)]",
        ]
    })
    sheets = {**sheets, "Readme": readme}

    with pd.ExcelWriter(path, engine="xlsxwriter",
                        engine_kwargs={"options": {"strings_to_formulas": False, "strings_to_urls": False}}) as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
            autosize(writer, name, df)


# ============================================================
# 10. CLI
# ============================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Observed-history error/stopping analysis: success == 1 only")
    p.add_argument("--input", type=Path, default=INPUT_FILE)
    p.add_argument("--sheet", default=SHEET_NAME)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--stokes-source", choices=["derive", "stored"], default=STOKES_SOURCE)
    p.add_argument("--no-plots", action="store_true")

    if argv is None:
        launcher = Path(sys.argv[0]) if sys.argv else Path("")
        if launcher.name in {"ipykernel_launcher.py", "ipykernel_launcher"} or (
            launcher.name == "__main__.py" and launcher.parent.name == "ipykernel"
        ): argv = []
    return p.parse_args(argv)


# ============================================================
# 11. Main: one-step complete analysis
# ============================================================
def main(argv=None):
    args = parse_args(argv)
    input_file = args.input.resolve()
    output_dir = (args.output_dir or input_file.parent / "observed_history_error_analysis").resolve()
    output_xlsx = output_dir / "observed_history_error_analysis.xlsx"

    if not input_file.exists(): raise FileNotFoundError(f"Input workbook not found: {input_file}")
    if input_file == output_xlsx: raise ValueError("Output must not overwrite input workbook")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Read + success == 1 ----------
    df = pd.read_csv(input_file) if input_file.suffix.lower() == ".csv" else pd.read_excel(input_file, sheet_name=args.sheet)
    df.columns = [str(c).strip() for c in df.columns]
    raw_input = df.copy()

    if "success" not in df: raise ValueError("Input workbook must contain a 'success' column.")
    success = pd.to_numeric(df["success"], errors="coerce")
    n_total = len(df)
    df = df.loc[success.eq(1)].copy().reset_index(drop=True)
    n_success = len(df)
    if df.empty: raise ValueError("No success == 1 rows found.")

    print(f"Success filter: {n_total} total -> {n_success} retained -> {n_total-n_success} excluded")

    # ---------- Validate ----------
    required = {"cycle", "target_color"}
    for pred, exp, _ in TARGETS.values(): required.update([pred, exp])
    if args.stokes_source == "derive": required -= {"pred_Stokes_shift_nm", "Stokes_shift_nm"}
    missing = sorted(required - set(df.columns))
    if missing: raise ValueError("Missing columns: " + ", ".join(missing))

    numeric = ["cycle"]
    for pred, exp, _ in TARGETS.values(): numeric += [pred, exp]
    numeric += [c for c in ["mass_1_g", "mass_2_g", "reagent_1_mL", "temperature", "time_h"] if c in df]

    for c in set(numeric):
        if c not in df: df[c] = np.nan
        old, new = df[c], pd.to_numeric(df[c], errors="coerce")
        if (old.notna() & new.isna()).any() or np.isinf(new.to_numpy(float)).any():
            raise ValueError(f"Invalid numeric values in {c}")
        df[c] = new

    if df[["cycle", "target_color"]].isna().any().any(): raise ValueError("Missing cycle or target_color.")
    if not ((df["cycle"] >= 1) & (df["cycle"] % 1 == 0)).all(): raise ValueError("Cycle must be positive integer.")

    df["cycle"] = df["cycle"].astype(int)
    df["target_color"] = df["target_color"].astype(str).str.strip().str.title()
    if not df["target_color"].isin(COLOR_WINDOWS).all():
        raise ValueError("target_color must be Blue, Green, Yellow or Orange.")

    # ---------- Precision + calculations ----------
    df = prepare_values(df.sort_values(["cycle", "target_color"]).reset_index(drop=True), args.stokes_source)
    per_formula = add_scores(add_errors(df))
    iteration = stopping_trace(per_formula)
    stopping = iteration.groupby("target_color", sort=True).tail(1).reset_index(drop=True)
    per_cycle = metrics_by_group(per_formula, ["cycle"])
    color_cycle = metrics_by_group(per_formula, ["target_color", "cycle"])
    cumulative = cumulative_metrics(per_formula)
    color_overall = metrics_by_group(per_formula, ["target_color"])
    overall = pd.DataFrame([group_metrics(per_formula)])
    hits = hit_summary(per_formula)
    quality = quality_summary(per_formula, n_total, n_success)

    # ---------- Metadata ----------
    metadata = {
        "input_file": str(input_file),
        "input_sha256": hashlib.sha256(input_file.read_bytes()).hexdigest(),
        "success_filter": {"criterion": "success == 1", "total": n_total, "retained": n_success},
        "precision": {
            "pred_Ex_max_nm": "ROUND_HALF_UP integer",
            "pred_Em_max_nm": "ROUND_HALF_UP integer",
            "pred_QY_%": "ROUND_HALF_UP 2 decimals",
            "measured_Em": "ROUND_HALF_UP integer",
            "measured_Ex_QY": "original precision",
            "Stokes": "round(raw Em - raw Ex) to integer"
        },
        "parameters": {
            "color_windows": COLOR_WINDOWS, "alpha": ALPHA, "softness_nm": COLOR_SOFTNESS_NM,
            "qy_bounds": QY_BOUNDS, "stokes_bounds": STOKES_BOUNDS,
            "abs_tolerance": ABS_TOLERANCE, "rel_tolerance": REL_TOLERANCE,
            "score_noise_floor": SCORE_NOISE_FLOOR, "patience_cycles": PATIENCE_CYCLES,
            "min_cycles": MIN_CYCLES, "max_cycles": MAX_CYCLES
        }
    }

    # ---------- Export everything ----------
    sheets = {
        "Iteration_scores": iteration,
        "Stopping_summary": stopping,
        "Per_formula_error": per_formula,
        "Per_cycle_metrics": per_cycle,
        "Color_cycle_metrics": color_cycle,
        "Color_cumulative": cumulative,
        "Color_overall": color_overall,
        "Overall_metrics": overall,
        "Hit_rate": hits,
        "Data_quality": quality,
        "Raw_input": raw_input,
    }
    export_results(output_xlsx, sheets, metadata)
    iteration.to_csv(output_dir / "iteration_scores.csv", index=False, encoding="utf-8-sig")
    (output_dir / "analysis_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not args.no_plots:
        try:
            plot_metrics(per_cycle, color_cycle, cumulative, output_dir)
        except ImportError as e:
            raise RuntimeError("Excel/CSV saved. Install matplotlib for plots or use --no-plots.") from e

    # ---------- Console ----------
    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETED")
    print("=" * 72)
    print(f"Input rows: {n_total} | success==1: {n_success} | excluded: {n_total-n_success}")
    print(f"Output: {output_xlsx}")

    print("\nOverall metrics:")
    print(overall.to_string(index=False))

    print("\nColor-specific metrics:")
    print(color_overall.to_string(index=False))

    print("\nStrict target-window hit rates:")
    print(hits.to_string(index=False))

    print("\nStopping summary:")
    cols = ["target_color", "cycle", "best_so_far_score", "patience_counter",
            "strict_hit_seen", "first_stop_cycle"]
    print(stopping[[c for c in cols if c in stopping]].to_string(index=False))


if __name__ == "__main__":
    main()