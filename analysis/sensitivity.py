"""Rescore the saved Ax-visited pools used for Figures S9 and S10."""
from importlib import import_module
import json
from pathlib import Path
from types import SimpleNamespace
import pandas as pd


def main():
    for name, folder in [
        ("weight_sensitivity_success_filtered_3runs", "ax_fpnn_catboost_weight_sensitivity_results"),
        ("weight_sensitivity_success_filtered_3runs", "ax_fpnn_catboost_weight_sensitivity_results_beta"),
    ]:
        module = import_module("scripts." + name)
        data = Path("results") / folder / "data"
        config = json.loads((data / "settings.json").read_text())
        settings = module.SensitivityCfg(**config["sensitivity"])
        pool = pd.read_csv(data / "shared_ax_visited_prediction_pool.csv")
        search = SimpleNamespace(**config["original_search"])
        summary, _, _, stability = module.analyze_shared_pool(pool, search, settings)
        output = Path("outputs") / folder
        output.mkdir(exist_ok=True, parents=True)
        runs = pd.read_csv(data / "independent_run_metrics.csv")
        aggregate = module.aggregate_independent_runs(runs)
        aggregate.to_csv(output / "weight_sensitivity_mean_sd_by_run.csv", index=False)
        summary.to_csv(output / "weight_sensitivity_summary.csv", index=False)
        stability.to_csv(output / "recommendation_stability.csv", index=False)


if __name__ == "__main__":
    main()

