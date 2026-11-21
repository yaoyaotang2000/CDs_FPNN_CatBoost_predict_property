"""Matched FPNN head comparison and fixed-budget success-gate sensitivity."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from importlib import import_module

train = import_module("scripts.02_train_fp_catboost_cycle")

OUT = Path("outputs/revision")
WINDOWS = {"Blue": (440, 480), "Green": (500, 540), "Yellow": (560, 580), "Orange": (580, 630)}


def inputs(raw):
    smiles = pd.read_excel("data/compound_smiles.xlsx")
    featurizer = train.CarbonDotFeaturizer(smiles, train.get_solvent_polarity_data())
    dataset = train.FPProcessDataset(raw, featurizer, has_y=False)
    batch = train.collate_fn([dataset[i] for i in range(len(dataset))])
    return [batch[k] for k in ("fps_1", "fps_2", "global_feat")]


def fit_nn(features, y, fit, test, seed, epochs, columns):
    train.set_seed(seed)
    net = train.FPProcessNN()
    if len(columns) == 1:
        net.head[-1] = torch.nn.Linear(64, 1)
    scaler = StandardScaler().fit(y[fit][:, columns])
    scaled = torch.tensor(scaler.transform(y[fit][:, columns]), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.0001)
    for _ in range(epochs):
        for indices in torch.randperm(len(fit), generator=generator).split(64):
            rows = fit[indices.numpy()]
            loss = torch.nn.functional.mse_loss(net(*(x[rows] for x in features)), scaled[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
    net.eval()
    with torch.no_grad():
        return scaler.inverse_transform(net(*(x[test] for x in features)).numpy())


def heads(raw):
    raw = raw.loc[raw.success.eq(1)].reset_index(names="source_row")
    y = raw[train.TARGET_NAMES].to_numpy()
    features = inputs(raw)
    rows = []
    for split, (fit, test) in enumerate(RepeatedKFold(n_splits=5, n_repeats=3, random_state=42).split(raw)):
        seed = 42 + split
        joint = fit_nn(features, y, fit, test, seed, 300, [0, 1, 2])
        for j, target in enumerate(train.TARGET_NAMES):
            separate = fit_nn(features, y, fit, test, seed, 300, [j]).ravel()
            for mode, pred in [("shared", joint[:, j]), ("separate", separate)]:
                for idx, value in zip(test, pred):
                    rows.append((split//5+1, split%5+1, raw.source_row[idx], target, mode, y[idx,j], value))
    data = pd.DataFrame(rows, columns=["repeat","fold","source_row","target","mode","measured","predicted"])
    data.to_csv(OUT/"head_predictions.csv", index=False)
    metrics = []
    for (rep, target, mode), g in data.groupby(["repeat","target","mode"]):
        metrics.append((rep,target,mode,r2_score(g.measured,g.predicted),mean_absolute_error(g.measured,g.predicted)))
    pd.DataFrame(metrics,columns=["repeat","target","mode","R2","MAE"]).to_csv(OUT/"head_metrics.csv",index=False)


def normalized(qy, shift):
    return np.clip(qy/100,0,1), np.clip((shift-40)/140,0,1)


def score(em, q, s, bounds):
    distance = np.maximum.reduce([bounds[0]-em, em-bounds[1], np.zeros(len(em))])
    return np.exp(-.5*(distance/10)**2) * (.7*q+.3*s)


def hv(q, s):
    order = np.argsort(q)
    return np.sum(np.diff(np.r_[0., q[order]]) * np.maximum.accumulate(s[order][::-1])[::-1])


def gate(raw):
    # Use only recipe-derived inputs. The fixed published feature lists carry development-set selection bias.
    features = inputs(raw)
    compounds, fps = train.build_compound_feature_table(pd.read_excel("data/compound_smiles.xlsx"))
    tab = train.build_dataset_features(raw, compounds, fps, has_targets=True)
    targets = train.TARGET_NAMES
    assert len(raw)==len(tab) and np.allclose(raw[targets],tab[targets], equal_nan=True)
    cols = {t:train.read_feature_list(path) for t,path in train.FEATURE_LIST_FILES.items()}
    scols = train.read_feature_list(train.Cfg().external_success_feature_list_path)
    y = raw[targets].to_numpy()
    success = raw.success.to_numpy().astype(bool)
    q,s = normalized(y[:,2], y[:,1]-y[:,0])
    results, decisions = [], []
    for repeat,(fit,test) in enumerate(StratifiedShuffleSplit(n_splits=3,test_size=.35,random_state=42).split(raw,success),1):
        good = fit[success[fit]]
        members, prob = [], []
        for seed in train.Cfg().seeds:
            args = dict(iterations=2000, learning_rate=.05, depth=4,l2_leaf_reg=5,
                        random_strength=1, bootstrap_type="Bernoulli", subsample=.9,
                        random_seed=seed, verbose=False, thread_count=3, allow_writing_files=False)
            clf = CatBoostClassifier(**args).fit(tab.iloc[fit][scols].fillna(0),success[fit].astype(int))
            prob.append(clf.predict_proba(tab.iloc[test][scols].fillna(0))[:,1])
            members.append(fit_nn(features,y,good,test,seed,300,[0,1,2]))
            preds = []
            for j,t in enumerate(targets):
                model = CatBoostRegressor(**args).fit(tab.iloc[good][cols[t]].fillna(0),y[good,j])
                preds.append(model.predict(tab.iloc[test][cols[t]].fillna(0)))
            members.append(np.column_stack(preds))
        p = np.mean(prob,axis=0)
        ex,em,qy = np.mean(members,axis=0).T
        sx,se,sq = np.std(members,axis=0).T
        qhat,shat = normalized(qy,em-ex)
        for color,bounds in WINDOWS.items():
            actual = score(y[test,1],q[test],s[test],bounds)
            high = success[test] & (actual>=np.quantile(actual[success[test]],.9))
            base = score(em,qhat,shat,bounds) + .2*score(em,sq/100,np.hypot(sx,se)/140,bounds)
            for method,rank,allowed in [("unweighted",base,np.ones(len(p),bool)),
                                        ("weighted",p*base,np.ones(len(p),bool)),("gate_050",p*base,p>=.5)]:
                eligible = np.flatnonzero(allowed)
                if len(eligible)<10:
                    raise ValueError("Fewer than ten eligible candidates")
                order = eligible[np.argsort(-rank[eligible],kind="stable")[:10]]
                observed = np.r_[good, test[order][success[test[order]]]]
                results.append((repeat,color,method,high[order].mean(),hv(q[observed],s[observed])-hv(q[good],s[good]),
                                high.sum(),(high & ~allowed).sum(),len(test),len(eligible)))
                for k,idx in enumerate(test):
                    decisions.append((repeat,color,method,idx,p[k],rank[k],bool(allowed[k]),bool(high[k]),k in order))
    pd.DataFrame(results,columns=["repeat","color","method","hit_rate","hv_gain","n_high","high_rejected","n_pool","eligible"]).to_csv(OUT/"filter_metrics.csv",index=False)
    pd.DataFrame(decisions,columns=["repeat","color","method","source_row","p_success","rank_score","eligible","high","selected"]).to_csv(OUT/"filter_decisions.csv",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", choices=["heads","gate"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    raw = pd.read_excel("data/output_with_molar_amount_pubchem.xlsx").rename(columns={"Mass1(g)":"mass_1_g","Mass2(g)":"mass_2_g"})
    {"heads": heads, "gate": gate}[args.analysis](raw)

