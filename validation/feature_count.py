
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURE_FILE = "./cd_cv_outputs/cd_hydrothermal_features.xlsx"
OUT_DIR = Path("./cd_catboost_feature_selection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5
N_REPEATS = 3
TOPK_LIST = [20,25,30,35,40, 50, 60]
TARGETS = ["Ex_max_nm", "Em_max_nm", "QY_%"]

def calc_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }

def load_feature_table(path):
    if str(path).lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)

def add_qy_extra_features(df):
    out = df.copy()
    def col(name, default=0.0):
        return out[name] if name in out.columns else default

    pair_total_hetero = col("pair_total_N") + col("pair_total_O") + col("pair_total_S") + col("pair_total_P")
    out["qy_pair_total_hetero"] = pair_total_hetero
    out["qy_pair_hetero_to_C"] = pair_total_hetero / (col("pair_total_C") + 1e-8)
    out["qy_acidity_x_temp"] = col("solv_relative_acidity_level") * col("temperature")
    out["qy_acidity_x_time"] = col("solv_relative_acidity_level") * col("time_h")
    out["qy_acidity_x_temp_time"] = col("solv_relative_acidity_level") * col("temp_time")
    out["qy_protic_x_temp_time"] = col("solv_is_protic") * col("temp_time")
    out["qy_acid_x_pair_total_N"] = col("solv_relative_acidity_level") * col("pair_total_N")
    out["qy_acid_x_pair_total_O"] = col("solv_relative_acidity_level") * col("pair_total_O")
    out["qy_acid_x_pair_total_S"] = col("solv_relative_acidity_level") * col("pair_total_S")
    out["qy_acid_x_pair_total_P"] = col("solv_relative_acidity_level") * col("pair_total_P")
    out["qy_temp_x_pair_total_hetero"] = col("temperature") * pair_total_hetero
    out["qy_time_x_pair_total_hetero"] = col("time_h") * pair_total_hetero
    out["qy_temp_time_x_pair_total_hetero"] = col("temp_time") * pair_total_hetero
    out["qy_acid_x_pair_N_to_C"] = col("solv_relative_acidity_level") * col("pair_N_to_C")
    out["qy_acid_x_pair_O_to_C"] = col("solv_relative_acidity_level") * col("pair_O_to_C")
    out["qy_acid_x_pair_S_to_C"] = col("solv_relative_acidity_level") * col("pair_S_to_C")
    out["qy_acid_x_pair_P_to_C"] = col("solv_relative_acidity_level") * col("pair_P_to_C")
    out["qy_acid_x_amine_plus_acid"] = col("solv_relative_acidity_level") * col("pair_amine_plus_acid")
    out["qy_protic_x_amine_plus_acid"] = col("solv_is_protic") * col("pair_amine_plus_acid")
    out["qy_temp_x_tanimoto"] = col("temperature") * col("fp_tanimoto")
    out["qy_time_x_tanimoto"] = col("time_h") * col("fp_tanimoto")
    out["qy_acid_x_tanimoto"] = col("solv_relative_acidity_level") * col("fp_tanimoto")
    out["qy_reagent_ratio_x_temp"] = col("reagent_to_precursor_vol_mass_ratio") * col("temperature")
    out["qy_reagent_ratio_x_time"] = col("reagent_to_precursor_vol_mass_ratio") * col("time_h")
    out["qy_reagent_ratio_x_hetero"] = col("reagent_to_precursor_vol_mass_ratio") * pair_total_hetero
    out["qy_mass_ratio_x_acidity"] = col("mass_ratio_12") * col("solv_relative_acidity_level")
    out["qy_mmol_ratio_x_acidity"] = col("pre_mmol_ratio_12") * col("solv_relative_acidity_level")
    return out

def prepare_data(df, target):
    meta_cols = [
        "precursor_1", "precursor_2", "reagent_1",
        "precursor_1_key", "precursor_2_key", "reagent_1_key",
        "p1_compound_name", "p1_compound_key", "p1_SMILES",
        "p2_compound_name", "p2_compound_key", "p2_SMILES",
    ]
    target_cols = ["Ex_max_nm", "Em_max_nm", "QY_%", "stoke_shift", "stoke_shift_calc_check"]

    work = df.copy()   
    work = add_qy_extra_features(work)
    work = work.dropna(subset=[target]).reset_index(drop=True)
    exclude_cols = set(meta_cols + target_cols)
    feature_cols = [c for c in work.columns if c not in exclude_cols]
    X = work[feature_cols].copy()
    y = pd.to_numeric(work[target], errors="coerce")
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)
    return work, X, y, feature_cols

def build_catboost(seed, iterations=400):
    return CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=iterations,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        random_strength=1.0,
        bootstrap_type="Bernoulli",
        subsample=0.9,
        grow_policy="SymmetricTree",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )

def get_global_importance(X, y, seed=42):
    model = build_catboost(seed=seed, iterations=300)
    model.fit(X, y, verbose=False)
    imp = model.get_feature_importance()
    return pd.DataFrame({"feature": X.columns, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)

def run_cv(X, y, target, ranked_features):
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    rows = []
    overall_rows = []
    for topk in TOPK_LIST:
        use_feats = ranked_features[:topk]
        Xk = X[use_feats].copy()
        oof = pd.DataFrame({"y_true": y.values, "y_pred": np.nan})
        for run_id, (tr_idx, te_idx) in enumerate(rkf.split(Xk), 1):
            rep = (run_id - 1)//N_SPLITS + 1
            fold = (run_id - 1)%N_SPLITS + 1
            X_tr, X_te = Xk.iloc[tr_idx], Xk.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            model = build_catboost(seed=SEED+run_id, iterations=500)
            model.fit(X_tr, y_tr, eval_set=(X_te, y_te), use_best_model=True, early_stopping_rounds=50, verbose=False)
            pred = model.predict(X_te)
            oof.loc[te_idx, "y_pred"] = pred 
            met = calc_metrics(y_te.values, pred)
            rows.append({"target": target, "topk": topk, "rep": rep, "fold": fold, **met})
        met = calc_metrics(oof["y_true"], oof["y_pred"])
        overall_rows.append({"target": target, "topk": topk, "OOF_R2": met["R2"], "OOF_RMSE": met["RMSE"], "OOF_MAE": met["MAE"]})
        oof.to_csv(OUT_DIR / f"{target}_top{topk}_oof.csv", index=False, encoding="utf-8-sig")
    return pd.DataFrame(rows), pd.DataFrame(overall_rows)

def summarize(df_long):
    g = df_long.groupby(["target", "topk"]).agg({"MAE":["mean","std"], "RMSE":["mean","std"], "R2":["mean","std"]}).reset_index()
    g.columns = ["target","topk","MAE_mean","MAE_std","RMSE_mean","RMSE_std","R2_mean","R2_std"]
    g["R2_mean±std"] = g.apply(lambda r: f"{r['R2_mean']:.3f} ± {r['R2_std']:.3f}", axis=1)
    g["RMSE_mean±std"] = g.apply(lambda r: f"{r['RMSE_mean']:.3f} ± {r['RMSE_std']:.3f}", axis=1)
    g["MAE_mean±std"] = g.apply(lambda r: f"{r['MAE_mean']:.3f} ± {r['MAE_std']:.3f}", axis=1)
    return g

def main():
    df = load_feature_table(FEATURE_FILE)
    all_best = []
    all_long = []
    all_overall = []
    for target in TARGETS:
        work, X, y, feature_cols = prepare_data(df, target)
        imp_df = get_global_importance(X, y, seed=SEED)
        imp_df["target"] = target
        imp_df.to_csv(OUT_DIR / f"{target}_global_importance.csv", index=False, encoding="utf-8-sig")
        ranked = imp_df["feature"].tolist()
        df_long, df_overall = run_cv(X, y, target, ranked)
        df_summary = summarize(df_long)
        df_long.to_csv(OUT_DIR / f"{target}_topk_repeat3x5_long.csv", index=False, encoding="utf-8-sig")
        df_summary.to_csv(OUT_DIR / f"{target}_topk_repeat3x5_summary.csv", index=False, encoding="utf-8-sig")
        df_overall.to_csv(OUT_DIR / f"{target}_topk_repeat3x5_oof_overall.csv", index=False, encoding="utf-8-sig")
        best = df_overall.sort_values(["OOF_R2","topk"], ascending=[False, True]).iloc[0]
        best_k = int(best["topk"])
        best_features = ranked[:best_k]
        feat_list_df = pd.DataFrame({
            "target": target,
            "rank": np.arange(1, best_k+1),
            "feature": best_features,
            "importance": imp_df.set_index("feature").loc[best_features, "importance"].values
        })
        feat_list_df.to_csv(OUT_DIR / f"{target}_best_top{best_k}_feature_list.csv", index=False, encoding="utf-8-sig")
        selected_df = work[["precursor_1","precursor_2","reagent_1", target] + best_features].copy()
        selected_df.to_csv(OUT_DIR / f"{target}_best_top{best_k}_selected_features.csv", index=False, encoding="utf-8-sig")
        all_best.append({
            "target": target,
            "best_topk": best_k,
            "OOF_R2": best["OOF_R2"],
            "OOF_RMSE": best["OOF_RMSE"],
            "OOF_MAE": best["OOF_MAE"],
            "selected_feature_csv": str(OUT_DIR / f"{target}_best_top{best_k}_selected_features.csv"),
            "feature_list_csv": str(OUT_DIR / f"{target}_best_top{best_k}_feature_list.csv"),
        })
        all_long.append(df_long)
        all_overall.append(df_overall)
    pd.concat(all_long, ignore_index=True).to_csv(OUT_DIR / "all_targets_topk_repeat3x5_long.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_overall, ignore_index=True).to_csv(OUT_DIR / "all_targets_topk_repeat3x5_oof_overall.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_best).to_csv(OUT_DIR / "all_targets_best_topk_selection.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(all_best).to_string(index=False))

if __name__ == "__main__":
    main()
