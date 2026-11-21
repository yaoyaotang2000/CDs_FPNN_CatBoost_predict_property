"""Recalculate validation statistics and revision figures from public data tables."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path("outputs/revision")
COLORS = {"Blue":"#32679c","Green":"#288368","Yellow":"#b99418","Orange":"#d66d36"}
WINDOWS = {"Blue":(440,480),"Green":(500,540),"Yellow":(560,580),"Orange":(580,630)}
TARGETS = ["Ex_max_nm","Em_max_nm","QY_%"]


def prospective():
    h = pd.read_csv("data/validation.csv")
    initial = pd.read_excel("data/output_with_molar_amount_pubchem.xlsx").query("success == 1")
    raw_estimates = h[["pred_"+t for t in TARGETS]].copy()
    for t in TARGETS:
        quantum = Decimal(".01" if t == "QY_%" else "1")
        h["pred_"+t] = h["pred_"+t].map(lambda x: float(Decimal(str(x)).quantize(quantum, rounding=ROUND_HALF_UP)))
    rows = []
    fig, axes = plt.subplots(1,3,figsize=(9,3.2),layout="constrained")
    for t,ax in zip(TARGETS,axes):
        a,b = h[t],h["pred_"+t]
        err = (a-b).abs()
        v = h[h.matched].copy()
        v["abs_error"] = (a-raw_estimates["pred_"+t]).abs()[v.index]
        rho = spearmanr(v["sd_"+t],v.abs_error).statistic
        rows.append((t,len(h),err.mean(),np.sqrt(np.mean(err**2)),a.corr(b),1-(err**2).sum()/((a-a.mean())**2).sum(),len(v),rho))
        for c,g in v.groupby("target_color"):
            ax.scatter(g["sd_"+t],g.abs_error,s=20,color=COLORS[c],label=c)
        unit = "pp" if t=="QY_%" else "nm"
        ax.set(title=f"{t.split('_')[0]}   ρ = {rho:.2f}",xlabel=f"Ensemble SD ({unit})",ylabel=f"Absolute error ({unit})")
    axes[0].legend(frameon=False,fontsize=7)
    fig.savefig(OUT/"dispersion_error.png",dpi=260)
    plt.close(fig)
    metrics = pd.DataFrame(rows,columns=["target","n","MAE","RMSE","Pearson_r","R2","matched_n","Spearman_rho"]).set_index("target")
    metrics.to_csv(OUT/"prospective_metrics.csv")
    h["hit"] = [WINDOWS[c][0]<=v<=WINDOWS[c][1] for c,v in zip(h.target_color,h.Em_max_nm)]
    stages = []
    for c, group in [("All", h), ("Green", h[h.target_color.eq("Green")])]:
        for early, g in group.groupby(group.cycle.le(5)):
            stages.append([c, "1–5" if early else "6–16", *[(g[t]-g["pred_"+t]).abs().mean() for t in TARGETS], len(g), g.hit.sum(), 100*g.hit.mean()])
    pd.DataFrame(stages,columns=["color","cycles",*TARGETS,"n","hits","percent"]).to_csv(OUT/"stage_metrics.csv",index=False)
    h.groupby("target_color").hit.agg(["sum","count","mean"]).to_csv(OUT/"strict_hits.csv")
    # Include panels a-c with measured QY and Stokes shift, and square parity panels d-e.
    fig=plt.figure(figsize=(10.4,6.2),layout="constrained")
    grid=fig.add_gridspec(2,6)
    a=fig.add_subplot(grid[0,:2])
    def frontier(d):
        p=d[["shift","QY_%"]].drop_duplicates().sort_values(["shift","QY_%"],ascending=False)
        return p.loc[p["QY_%"].gt(p["QY_%"].cummax().shift(fill_value=-1))].sort_values("shift")
    initial=initial.assign(shift=initial.Stokes_shift_nm)
    h["shift"]=h.Em_max_nm-h.Ex_max_nm
    a.scatter(initial["shift"],initial["QY_%"],s=6,color="#c0c7cb")
    for d,c,name in [(initial,"#516171","Initial"),(pd.concat([initial,h]),"#a64347","Cumulative")]:
        p=frontier(d)
        a.step(p["shift"],p["QY_%"],where="post",color=c,label=name)
        a.scatter(p["shift"],p["QY_%"],s=10,color=c)
    a.set(xlabel="Stokes shift (nm)",ylabel="Measured QY (%)",title="(a) Measured Pareto fronts")
    a.legend(fontsize=7,frameon=False)
    bgrid=grid[0,2:].subgridspec(2,1,height_ratios=[1,2],hspace=.05)
    b=fig.add_subplot(bgrid[0])
    rmse=(h.pred_Em_max_nm-h.Em_max_nm).pow(2).groupby(h.cycle).mean().pow(.5)
    b.plot(rmse.index,rmse,color="#404950",marker=".",ms=3)
    b.set(ylabel="RMSE (nm)",title="(b) Prospective emission by cycle")
    b.tick_params(labelbottom=False)
    b2=fig.add_subplot(bgrid[1],sharex=b)
    for color,g in h.groupby("target_color"):
        b2.scatter(g.cycle,g.Em_max_nm,c=COLORS[color],s=14)
        b2.scatter(g.cycle,g.pred_Em_max_nm,edgecolors=COLORS[color],facecolors="none",marker="s",s=14)
    b2.set(xlabel="Cycle",ylabel="Emission (nm)")
    c=fig.add_subplot(grid[1,:2])
    hv=pd.read_csv("results/pareto_hv_results/pareto_hv_summary.csv").set_index("target_color").loc[list(COLORS)]
    x=np.arange(4)
    c.bar(x-.17,hv.baseline_hv,width=.34,color="#a9babf",label="Initial")
    c.bar(x+.17,hv.final_hv,width=.34,color="#4c6770",label="Cumulative")
    c.set(xticks=x,xticklabels=list(COLORS),ylabel="Hypervolume",title="(c) In-window coverage")
    c.legend(frameon=False,fontsize=7)
    for ax,t,label,title in [(fig.add_subplot(grid[1,2:4]),"Em_max_nm","Emission (nm)","(d) Emission"),(fig.add_subplot(grid[1,4:]),"QY_%","QY (%)","(e) Quantum yield")]:
        bounds=h[[t,"pred_"+t]].to_numpy()
        low,high=np.floor(bounds.min()/10)*10,np.ceil(bounds.max()/10)*10
        ax.plot([low,high],[low,high],ls="--",c="#8c9395",lw=1)
        for color,g in h.groupby("target_color"):
            ax.scatter(g[t],g["pred_"+t],label=color,color=COLORS[color],s=16)
        ax.set(xlim=(low,high),ylim=(low,high),aspect="equal",xlabel="Measured "+label,ylabel="Predicted "+label,title=title)
        ax.text(.04,.96,f"MAE {metrics.loc[t,'MAE']:.2f}\nr {metrics.loc[t,'Pearson_r']:.2f}   R² {metrics.loc[t,'R2']:.2f}",va="top",transform=ax.transAxes,fontsize=8)
    fig.savefig(OUT/"figure3.png",dpi=320)
    plt.close(fig)


def retrospective():
    t = pd.read_csv("results/ensemble_fpnn_catboost_reviewer_results/data/benchmark_trajectory.csv")
    keys=["task","repeat","strategy"]
    initial=t.query("step == 0").set_index(keys)
    final=t.query("step == 10").set_index(keys).copy()
    assert initial.index.is_unique and final.index.is_unique and len(final)==4*20*6
    assert initial.groupby(["task","repeat"]).hypervolume.nunique().max()==1
    final["hv_gain"]=final.hypervolume-initial.hypervolume
    final.reset_index().to_csv(OUT/"benchmark_by_repeat.csv",index=False)
    final.groupby(["task","strategy"])[["selected_only_hit_rate_top10","hv_gain"]].agg(["mean","std"]).to_csv(OUT/"benchmark_by_task.csv")
    pivot=final.selected_only_hit_rate_top10.unstack()
    pivot["UCB_minus_greedy_pp"]=(pivot.Ensemble_UCB-pivot.Ensemble_greedy)*100
    pivot.to_csv(OUT/"hit_pairs.csv")
    names=["Random","Chemistry_heuristic","CatBoost_greedy","FPProcessNN_greedy","Ensemble_greedy","Ensemble_UCB"]
    fig,axes=plt.subplots(1,3,figsize=(10,3.4),layout="constrained")
    for i,(ax,col,caption) in enumerate(zip(axes,["selected_only_hit_rate_top10","mean_score_newly_selected","selected_only_hit_rate_top10"],["(a) Top-decile hits","(b) Measured objective","(c) Across-color hits"])):
        vals=final[col].reset_index()
        if i<2:
            for j,color in enumerate(COLORS):
                g=vals[vals.task==color].groupby("strategy")[col].agg(["mean","std"]).loc[names]
                ax.errorbar(np.arange(6)+(j-1.5)*.12,g["mean"],yerr=g["std"],fmt=".",capsize=2,color=COLORS[color],label=color)
        else:
            g=vals.groupby(["repeat","strategy"])[col].mean().groupby("strategy").agg(["mean","std"]).loc[names]
            ax.bar(np.arange(6),g["mean"],yerr=g["std"],color="#628293",capsize=2)
        ax.set(xticks=range(6),xticklabels=list("ABCDEF"),title=caption,ylabel="Fraction" if i!=1 else "Score")
    axes[0].legend(fontsize=7,frameon=False)
    fig.savefig(OUT/"figure_s11.png",dpi=260)
    plt.close(fig)


if __name__=="__main__":
    OUT.mkdir(exist_ok=True,parents=True)
    prospective()
    retrospective()

