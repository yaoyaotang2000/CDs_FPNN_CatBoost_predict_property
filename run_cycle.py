"""Run the notebook-derived closed loop from the repository root."""
import argparse
from pathlib import Path
from importlib import import_module

from workflow.stages import prepare_cycle, train_cycle, search_cycle


def example():
    search = import_module("scripts.03_ax_bo_search_cycle")
    root = next(Path("checkpoints").glob("cycle_01_*"))
    model = search.EnsemblePredictor(search.Cfg(run_dir=str(root),cycle=1,exclude_tested_recipes=False))
    assert len(model.nn_models)==5 and len(model.success_models)==5
    assert all(len(group)==5 for group in model.cb_models.values())
    recipe = dict(precursor_1="DL-Methionine",precursor_2="2-Aminoterephthalic acid",mass_1_g=.8,mass_2_g=.8,
                  reagent_1="water",temperature=160,time_h=4)
    pred = model.predict(recipe)
    return {k: pred[k] for k in ("Ensemble_pred_Ex_max_nm","Ensemble_pred_Em_max_nm","Ensemble_pred_QY_%","P_success")}


def run(cycle=1, history=None, quick=False, prepare_only=False):
    path = prepare_cycle(cycle, history)
    if not prepare_only:
        train_cycle(path, cycle, quick)
        search_cycle(path, cycle, quick)
    return path


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=["predict","prepare","quick","cycle"])
    parser.add_argument("--cycle",type=int,default=1)
    parser.add_argument("--history",type=Path)
    a=parser.parse_args()
    print(example() if a.action=="predict" else run(a.cycle,a.history,a.action=="quick",a.action=="prepare"))

