"""Handoff between the three notebook-derived scripts."""
import argparse
from importlib import import_module
import json
from pathlib import Path

OUTPUTS = Path("outputs/closed_loop_runs")
LOCAL_HISTORY = Path("data/observed_history.xlsx")


def prepare_cycle(cycle, history=None):
    if cycle < 1:
        raise ValueError("Cycle numbers start at 1.")
    path = Path(history) if history is not None else LOCAL_HISTORY
    if (history is not None or cycle > 1) and not path.is_file():
        raise FileNotFoundError(f"Provide the measured history for cycle {cycle}: {path}")
    module = import_module("scripts.01_update_training_data_from_history")
    cfg = module.Cfg(cycle=cycle, observed_history_file=str(path), base_run_dir=str(OUTPUTS))
    return module.main(cfg)


def resolve_run(run_dir=None, cycle=None):
    if run_dir is None:
        prefix = "*" if cycle is None else f"{cycle:02d}"
        runs = [p for p in OUTPUTS.glob(f"cycle_{prefix}_*") if (p / "data/run_manifest.json").is_file()]
        if not runs:
            raise FileNotFoundError("Run scripts/01_update_training_data_from_history.py first.")
        run_dir = max(runs, key=lambda p: p.name.split("_", 2)[-1])
    path = Path(run_dir)
    manifest = json.loads((path / "data/run_manifest.json").read_text(encoding="utf-8"))
    n = int(manifest["cycle"])
    if cycle is not None and cycle != n:
        raise ValueError(f"Selected directory is cycle {n}, requested {cycle}.")
    return path, n


def train_cycle(path, cycle, quick=False):
    module = import_module("scripts.02_train_fp_catboost_cycle")
    if any((path / "models").iterdir()):
        raise FileExistsError("This run already contains fitted or partial models. Start a fresh run with step 01.")
    cfg = module.Cfg(cycle=cycle, run_dir=str(path))
    if quick:
        cfg.seeds = (42,)
        cfg.nn_epochs = 1
        cfg.cb_iterations = 5
    module.main(cfg)


def search_cycle(path, cycle, quick=False):
    module = import_module("scripts.03_ax_bo_search_cycle")
    fit = json.loads((path / "data/train_config.json").read_text(encoding="utf-8"))
    model_dir = path / "models"
    required = ["y_scaler.pkl"]
    for seed in fit["seeds"]:
        required.append(f"FPProcessNN_seed{seed}.pt")
        required.extend(f"catboost_{target}_seed{seed}.cbm" for target in (*module.TARGET_NAMES,"success"))
    for name in required:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    for name in ("prior_history.csv","success_classifier_feature_list.csv"):
        if not (path / "data" / name).is_file():
            raise FileNotFoundError(path / "data" / name)
    cfg = module.Cfg(cycle=cycle, run_dir=str(path), observed_history_file=str(path / "data/prior_history.csv"))
    if quick:
        cfg.runs_per_color = 1
        cfg.total_trials = 2
    module.main(cfg)


def stage_cli(stage):
    labels = {1:"Prepare data available before the requested cycle.",2:"Fit the neural and CatBoost ensembles.",3:"Search using fitted ensemble members."}
    parser = argparse.ArgumentParser(description=labels[stage])
    parser.add_argument("--cycle",type=int,default=1 if stage==1 else None)
    if stage==1:
        parser.add_argument("--history",type=Path,help="Local measurement table, CSV or XLSX.")
    else:
        parser.add_argument("--run-dir",type=Path,help="Prepared run; defaults to the most recently created matching run.")
        parser.add_argument("--quick",action="store_true",help="Use a reduced demonstration budget.")
    args = parser.parse_args()
    if stage==1:
        print(prepare_cycle(args.cycle,args.history))
    else:
        path, cycle = resolve_run(args.run_dir,args.cycle)
        (train_cycle if stage==2 else search_cycle)(path,cycle,args.quick)
        print(path)

