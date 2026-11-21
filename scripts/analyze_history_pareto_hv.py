#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pareto-front and hypervolume analysis for closed-loop AL/BO experiments.

Primary objectives (both maximized): experimentally measured QY (%) and
Stokes shift (nm). Model predictions are never used to assess optimization.

Examples
--------
# History-only provisional analysis (cycle 1 is the baseline):
python analyze_history_pareto_hv.py --history observed_history.xlsx

# Manuscript-ready comparison using the true initialization experiments:
python analyze_history_pareto_hv.py --history observed_history.xlsx \
    --initial initial_training_data.xlsx

Unsuccessful experiments are retained in counts/success rates, but cannot be
feasible Pareto points when valid optical measurements are unavailable.
"""

from __future__ import annotations

# Allow the documented "python scripts/..." entry point from a checkout.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLORS = {
    "Blue": "#2878B5",
    "Green": "#2A9D67",
    "Yellow": "#E3A008",
    "Orange": "#D55E00",
}
COLOR_WINDOWS = {
    "Blue": (440.0, 480.0),
    "Green": (500.0, 540.0),
    "Yellow": (560.0, 580.0),
    "Orange": (580.0, 630.0),
}
ITER_CMAP = "viridis"
QY_BOUNDS = (0.0, 100.0)
STOKES_BOUNDS = (40.0, 180.0)
REFERENCE_POINT = (0.0, 0.0)


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    return pd.read_csv(p)


def standardize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    d = df.copy()
    aliases = {
        "stokes_shift": "Stokes_shift_nm",
        "stoke_shift": "Stokes_shift_nm",
        "QY": "QY_%",
        "qy": "QY_%",
        "emission": "Em_max_nm",
        "excitation": "Ex_max_nm",
        "Mass1(g)": "mass_1_g",
        "Mass2(g)": "mass_2_g",
    }
    for old, new in aliases.items():
        if old in d.columns and new not in d.columns:
            d = d.rename(columns={old: new})
    if "Stokes_shift_nm" not in d.columns and {"Em_max_nm", "Ex_max_nm"}.issubset(d.columns):
        d["Stokes_shift_nm"] = d["Em_max_nm"] - d["Ex_max_nm"]
    if "success" not in d.columns:
        d["success"] = 1
    if "cycle" not in d.columns:
        d["cycle"] = 0 if source == "initialization" else np.nan
    if "target_color" not in d.columns:
        d["target_color"] = "Unassigned"
    d["data_source"] = source
    d["cycle"] = pd.to_numeric(d["cycle"], errors="coerce")
    d["success"] = pd.to_numeric(d["success"], errors="coerce").fillna(0).astype(int)
    for c in ["QY_%", "Stokes_shift_nm", "Ex_max_nm", "Em_max_nm"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    required = ["QY_%", "Stokes_shift_nm"]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing measured objective columns: {missing}")
    return d


def subset_for_color(d: pd.DataFrame, color: str) -> pd.DataFrame:

    lo, hi = COLOR_WINDOWS[color]
    hit = d["Em_max_nm"].between(lo, hi, inclusive="both") if "Em_max_nm" in d else pd.Series(False, index=d.index)
    init = d["data_source"].eq("initialization")
    post = d["data_source"].eq("observed_history") & d["target_color"].astype(str).str.casefold().eq(color.casefold())
    return d.loc[hit & (init | post)].copy()


def feasible(d: pd.DataFrame) -> pd.DataFrame:
    return d.loc[
        d["success"].eq(1)
        & d["QY_%"].notna()
        & d["Stokes_shift_nm"].notna()
    ].copy()


def add_recipe_keys(initial: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add reproducible recipe keys and flags for repeated recommendations."""
    cols = ["precursor_1", "precursor_2", "reagent_1", "temperature", "time_h",
            "mass_1_g", "mass_2_g", "pre_1_mmol", "pre_2_mmol"]

    def keyed(d: pd.DataFrame) -> pd.DataFrame:
        x = d.copy()
        parts = []
        for c in cols:
            if c in x:
                s = x[c]
                if pd.api.types.is_numeric_dtype(s):
                    s = pd.to_numeric(s, errors="coerce").round(6).astype("string")
                else:
                    s = s.astype("string").str.strip().str.casefold()
            else:
                s = pd.Series("<missing>", index=x.index, dtype="string")
            parts.append(s.fillna("<na>"))
        key = parts[0]
        for s in parts[1:]:
            key = key.str.cat(s, sep="|")
        x["recipe_key"] = key
        return x

    ini, hist = keyed(initial), keyed(history)
    prior = set(ini["recipe_key"].dropna())
    hist["duplicate_of_initial_recipe"] = hist["recipe_key"].isin(prior)
    hist["duplicate_within_history"] = hist.duplicated("recipe_key", keep=False)
    ini["duplicate_of_initial_recipe"] = False
    ini["duplicate_within_history"] = False
    return ini, hist


def normalize_objectives(d: pd.DataFrame) -> np.ndarray:
    q = (d["QY_%"].to_numpy(float) - QY_BOUNDS[0]) / (QY_BOUNDS[1] - QY_BOUNDS[0])
    s = (d["Stokes_shift_nm"].to_numpy(float) - STOKES_BOUNDS[0]) / (STOKES_BOUNDS[1] - STOKES_BOUNDS[0])
    return np.column_stack([np.clip(s, 0, 1), np.clip(q, 0, 1)])


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    """Return mask of non-dominated points for a maximization problem."""
    n = len(points)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        dominates_i = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        if np.any(dominates_i):
            keep[i] = False
    return keep


def pareto_frame(d: pd.DataFrame) -> pd.DataFrame:
    f = feasible(d)
    if f.empty:
        return f
    m = nondominated_mask(normalize_objectives(f))
    # Identical objective pairs occupy the same location and contribute only
    # once to the front/HV. Keep the first record for traceability.
    return (f.loc[m]
            .sort_values(["Stokes_shift_nm", "QY_%", "cycle"])
            .drop_duplicates(subset=["Stokes_shift_nm", "QY_%"], keep="first")
            .copy())


def hypervolume_2d(d: pd.DataFrame) -> float:
    """Exact 2-D hypervolume for maximization, normalized bounds, r=(0,0)."""
    p = pareto_frame(d)
    if p.empty:
        return 0.0
    xy = normalize_objectives(p)
    xy = xy[np.argsort(xy[:, 0])]
    hv, prev_x = 0.0, REFERENCE_POINT[0]
    for x, y in xy:
        if x > prev_x:
            hv += (x - prev_x) * max(y - REFERENCE_POINT[1], 0.0)
            prev_x = x
    return float(hv)


def bootstrap_hv(d: pd.DataFrame, n_boot: int, seed: int) -> tuple[float, float, float]:
    f = feasible(d)
    if f.empty:
        return 0.0, 0.0, 0.0
    points = normalize_objectives(f)

    def hv_array(xy: np.ndarray) -> float:
        # Sort by Stokes coordinate; for duplicate x retain the largest QY.
        order = np.lexsort((-xy[:, 1], xy[:, 0]))
        z = xy[order]
        _, first = np.unique(z[:, 0], return_index=True)
        z = z[np.sort(first)]
        # Only points with a QY larger than every point to their right matter.
        suffix_max = np.maximum.accumulate(z[::-1, 1])[::-1]
        keep = z[:, 1] >= suffix_max - 1e-15
        z = z[keep]
        widths = np.diff(np.r_[REFERENCE_POINT[0], z[:, 0]])
        return float(np.sum(np.maximum(widths, 0.0) * np.maximum(z[:, 1] - REFERENCE_POINT[1], 0.0)))

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(points), len(points))
        vals[i] = hv_array(points[idx])
    return hv_array(points), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def cumulative_sets(initial: pd.DataFrame, history: pd.DataFrame, baseline_mode: str):
    cycles = sorted(int(x) for x in history["cycle"].dropna().unique())
    if baseline_mode == "true_initialization":
        yield "Initial", 0, initial.copy()
        base = initial.copy()
        for c in cycles:
            base = pd.concat([base, history.loc[history["cycle"].eq(c)]], ignore_index=True, sort=False)
            yield f"Cycle {c}", c, base.copy()
    else:
        first = cycles[0]
        base = history.loc[history["cycle"].eq(first)].copy()
        yield f"Cycle {first} baseline", first, base.copy()
        for c in cycles[1:]:
            base = pd.concat([base, history.loc[history["cycle"].eq(c)]], ignore_index=True, sort=False)
            yield f"Cycle {c}", c, base.copy()


def build_tables(initial: pd.DataFrame, history: pd.DataFrame, baseline_mode: str, n_boot: int, seed: int):
    all_rows = pd.concat([initial, history], ignore_index=True, sort=False) if len(initial) else history.copy()
    all_rows["feasible_for_pareto"] = (
        all_rows["success"].eq(1) & all_rows["QY_%"].notna() & all_rows["Stokes_shift_nm"].notna()
    )
    all_rows["is_pareto_final_pooled"] = False
    pf_final = pareto_frame(all_rows)
    all_rows.loc[pf_final.index.intersection(all_rows.index), "is_pareto_final_pooled"] = True

    hv_rows = []
    cumulative_cache = list(cumulative_sets(initial, history, baseline_mode))
    for j, (label, cyc, d) in enumerate(cumulative_cache):
        scopes = [("All", d)] + [(color, subset_for_color(d, color)) for color in COLORS]
        for k, (color, sub) in enumerate(scopes):
            hv, lo, hi = bootstrap_hv(sub, n_boot=n_boot, seed=seed + j * 97 + k)
            pf = pareto_frame(sub)
            hv_rows.append({
                "stage": label, "cycle": cyc, "target_color": color,
                "n_total": len(sub), "n_success": int(sub["success"].eq(1).sum()),
                "success_rate": float(sub["success"].mean()) if len(sub) else np.nan,
                "n_feasible": len(feasible(sub)), "n_pareto": len(pf),
                "hypervolume": hv, "hv_ci_low": lo, "hv_ci_high": hi,
            })
    hv_df = pd.DataFrame(hv_rows)
    hv_df["hv_increment"] = hv_df.groupby("target_color")["hypervolume"].diff().fillna(0.0)

    first_label, _, first_set = cumulative_cache[0]
    _, _, final_set = cumulative_cache[-1]
    summary_rows = []
    colors = ["All"] + list(COLORS)
    for color in colors:
        b = first_set if color == "All" else subset_for_color(first_set, color)
        f = final_set if color == "All" else subset_for_color(final_set, color)
        hb, lb, ub = bootstrap_hv(b, n_boot, seed + 1000)
        hf, lf, uf = bootstrap_hv(f, n_boot, seed + 2000)
        summary_rows.append({
            "target_color": color, "baseline_label": first_label,
            "baseline_n": len(b), "final_n": len(f),
            "baseline_pareto_n": len(pareto_frame(b)), "final_pareto_n": len(pareto_frame(f)),
            "baseline_hv": hb, "baseline_hv_ci_low": lb, "baseline_hv_ci_high": ub,
            "final_hv": hf, "final_hv_ci_low": lf, "final_hv_ci_high": uf,
            "absolute_hv_gain": hf - hb,
            "relative_hv_gain_pct": (hf / hb - 1) * 100 if hb > 0 else np.nan,
        })
    return all_rows, hv_df, pd.DataFrame(summary_rows), first_set, final_set


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#D9DEE7", linewidth=0.65, alpha=0.7)
    ax.set_axisbelow(True)


def draw_front(ax, p: pd.DataFrame, color: str, label: str, ls: str = "-"):
    if p.empty:
        return
    x = p["Stokes_shift_nm"].to_numpy()
    y = p["QY_%"].to_numpy()
    # Staircase boundary for simultaneous maximization.
    ax.step(x, y, where="post", color=color, lw=2.2, ls=ls, label=label, zorder=5)


def plot_pooled(first_set, final_set, baseline_mode: str, outdir: Path):
    fig, ax = plt.subplots(figsize=(7.6, 6.0), constrained_layout=True)
    pf0, pff = pareto_frame(first_set), pareto_frame(final_set)
    ax.scatter(first_set["Stokes_shift_nm"], first_set["QY_%"], s=54, c="#AAB2BF", alpha=.85,
               edgecolor="white", linewidth=.6, label="Baseline experiments", zorder=2)
    hist_only = final_set.loc[final_set["data_source"].eq("observed_history")]
    sc = ax.scatter(hist_only["Stokes_shift_nm"], hist_only["QY_%"], s=62,
                    c=hist_only["cycle"], cmap=ITER_CMAP, edgecolor="white", linewidth=.7,
                    alpha=.92, label="Later AL experiments", zorder=3)
    draw_front(ax, pf0, "#68707D", "Baseline Pareto front", "--")
    draw_front(ax, pff, "#B2182B", "Final cumulative Pareto front", "-")
    ax.scatter(pff["Stokes_shift_nm"], pff["QY_%"], marker="*", s=165, facecolor="#B2182B",
               edgecolor="white", linewidth=.8, zorder=6)
    ax.set_xlabel("Measured Stokes shift (nm)")
    ax.set_ylabel("Measured quantum yield, QY (%)")
    ax.set_title("Experimentally observed Pareto-front expansion", loc="left", weight="bold", pad=12)
    subtitle = "Cycle 1 is a provisional baseline; true initialization data were not supplied" if baseline_mode != "true_initialization" else "True initialization experiments versus cumulative post-optimization observations"
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.2, color="#5F6772")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    cbar = fig.colorbar(sc, ax=ax, pad=.02, fraction=.045)
    cbar.set_label("AL cycle")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(outdir / f"Fig1_pooled_pareto_front.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_hv(hv_df: pd.DataFrame, baseline_mode: str, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    overall = hv_df.loc[hv_df["target_color"].eq("All")].sort_values("cycle")
    x = np.arange(len(overall))
    axes[0].plot(x, overall["hypervolume"], color="#1F6F8B", marker="o", lw=2.4)
    axes[0].fill_between(x, overall["hv_ci_low"], overall["hv_ci_high"], color="#1F6F8B", alpha=.18, linewidth=0)
    axes[0].set_xticks(x, overall["cycle"].astype(int))
    axes[0].set_xlabel("AL cycle")
    axes[0].set_ylabel("Normalized hypervolume")
    axes[0].set_title("a  Cumulative hypervolume", loc="left", weight="bold")
    style_axes(axes[0])

    for color in [c for c in COLORS if c in set(hv_df["target_color"])]:
        g = hv_df.loc[hv_df["target_color"].eq(color)].sort_values("cycle")
        axes[1].plot(g["cycle"], g["hypervolume"], marker="o", ms=4.5, lw=1.8,
                     color=COLORS[color], label=color)
    axes[1].set_xlabel("AL cycle")
    axes[1].set_ylabel("Normalized hypervolume")
    axes[1].set_title("b  Color-resolved hypervolume", loc="left", weight="bold")
    style_axes(axes[1])
    axes[1].legend(frameon=False, ncol=2)
    note = (f"Shading: bootstrap 95% CI. Fixed bounds: QY {QY_BOUNDS[0]:g}–{QY_BOUNDS[1]:g}%; "
            f"Stokes shift {STOKES_BOUNDS[0]:g}–{STOKES_BOUNDS[1]:g} nm; reference=(0,0).")
    fig.text(.5, -.025, note, ha="center", fontsize=9, color="#5F6772")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(outdir / f"Fig2_hypervolume_by_cycle.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_color_fronts(first_set, final_set, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.2), sharex=True, sharey=True, constrained_layout=True)
    for ax, color in zip(axes.flat, COLORS):
        b = subset_for_color(first_set, color)
        f = subset_for_color(final_set, color)
        later = f.loc[f["data_source"].eq("observed_history")]
        ax.scatter(b["Stokes_shift_nm"], b["QY_%"], s=65, c="#AAB2BF", edgecolor="white", linewidth=.6, zorder=2)
        ax.scatter(later["Stokes_shift_nm"], later["QY_%"], s=58, c=COLORS[color], alpha=.82,
                   edgecolor="white", linewidth=.6, zorder=3)
        draw_front(ax, pareto_frame(b), "#68707D", "Baseline", "--")
        draw_front(ax, pareto_frame(f), COLORS[color], "Final cumulative", "-")
        pf = pareto_frame(f)
        ax.scatter(pf["Stokes_shift_nm"], pf["QY_%"], marker="*", s=130,
                   c=COLORS[color], edgecolor="white", linewidth=.7, zorder=5)
        ax.set_title(color, weight="bold", color=COLORS[color])
        style_axes(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Measured Stokes shift (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Measured QY (%)")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#AAB2BF", markeredgecolor="white", label="Baseline experiments", markersize=8),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#2878B5", markeredgecolor="white", label="Later AL experiments", markersize=8),
        Line2D([0], [0], color="#68707D", ls="--", lw=2, label="Baseline Pareto front"),
        Line2D([0], [0], color="#B2182B", lw=2, label="Final cumulative Pareto front"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Color-resolved experimental Pareto fronts", weight="bold", fontsize=14)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(outdir / f"Fig3_color_resolved_pareto.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="./data/validation.csv", help="observed_history.xlsx/csv")
    ap.add_argument("--initial", default="./data/output_with_molar_amount_pubchem.xlsx", help="true initialization dataset")
    ap.add_argument("--outdir", default="outputs/pareto")
    ap.add_argument("--bootstrap", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    history = standardize(
        read_table(args.history),"observed_history")
    if args.initial:

        initial_raw = read_table(args.initial)
        if "success" not in initial_raw.columns:
            raise ValueError(
                f"The initialization dataset {args.initial} does not contain "
                "a 'success' column. A success column is required because "
                "only success == 1 samples should be used."
            )
        success_numeric = pd.to_numeric(initial_raw["success"], errors="coerce")
        n_initial_all = len(initial_raw)

        initial_raw = initial_raw.loc[success_numeric.eq(1)].copy()
        n_initial_success = len(initial_raw)

        print(
            f"Initialization dataset: "
            f"{n_initial_all} total rows -> "
            f"{n_initial_success} rows with success == 1 retained."
        )
        # Standardize only after filtering
        initial = standardize(
            initial_raw,
            "initialization")
        baseline_mode = "true_initialization"

    else:
        initial = pd.DataFrame(columns=history.columns)
        baseline_mode = "cycle_1_provisional"

    if {"precursor_1", "precursor_2"}.issubset(history.columns):
        initial, history = add_recipe_keys(initial, history)

    rows, hv, summary, first_set, final_set = build_tables(initial, history, baseline_mode, args.bootstrap, args.seed)
    rows.to_csv(outdir / "processed_experiments.csv", index=False, encoding="utf-8-sig")
    hv.to_csv(outdir / "hypervolume_by_cycle.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(outdir / "pareto_hv_summary.csv", index=False, encoding="utf-8-sig")
    pareto_frame(first_set).to_csv(outdir / "baseline_pareto_points.csv", index=False, encoding="utf-8-sig")
    pareto_frame(final_set).to_csv(outdir / "final_cumulative_pareto_points.csv", index=False, encoding="utf-8-sig")

    plot_pooled(first_set, final_set, baseline_mode, outdir)
    plot_hv(hv, baseline_mode, outdir)
    plot_color_fronts(first_set, final_set, outdir)

    manifest = {
        "history_file": str(Path(args.history).resolve()),
        "initial_file": str(Path(args.initial).resolve()) if args.initial else None,
        "baseline_mode": baseline_mode,
        "warning": None if args.initial else "Cycle 1 is only a provisional baseline; supply the true initialization dataset for the reviewer comparison.",
        "objectives": ["measured Stokes_shift_nm (maximize)", "measured QY_% (maximize)"],
        "normalization_bounds": {"QY_%": QY_BOUNDS, "Stokes_shift_nm": STOKES_BOUNDS},
        "reference_point_normalized": REFERENCE_POINT,
        "color_windows_nm_inclusive": COLOR_WINDOWS,
        "color_scope_rule": "Initialization: measured Em in window. History: intended target_color and measured Em in the same window.",
        "bootstrap_replicates": args.bootstrap,
        "seed": args.seed,
    }
    (outdir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"\nResults saved to {outdir.resolve()}")


if __name__ == "__main__":
    main()
