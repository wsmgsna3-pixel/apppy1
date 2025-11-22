# optimize.py
# Optuna wrapper for parameter optimization (uses backtrader runner in ycjsb_backtest_addon)

import optuna
import json
import os
from functools import partial
from typing import Dict, Any
from tqdm import tqdm

def objective_factory(runner_func, fixed_args):
    """
    runner_func: function that accepts params dict and returns metric (higher better)
    fixed_args: other args passed in
    """
    def objective(trial):
        # example tunables
        stoploss = trial.suggest_uniform("stoploss_pct", 0.02, 0.12)
        takeprofit = trial.suggest_uniform("takeprofit_pct", 0.05, 0.4)
        vol_ratio_min = trial.suggest_uniform("vol_ratio_min", 1.0, 2.5)
        macd_min = trial.suggest_uniform("macd_min", -0.5, 0.5)
        rsi_max = trial.suggest_uniform("rsi_max", 60, 85)
        params = {
            "stoploss_pct": stoploss,
            "takeprofit_pct": takeprofit,
            "VOL_RATIO_MIN": vol_ratio_min,
            "MACD_MIN": macd_min,
            "RSI_MAX": rsi_max
        }
        metric = runner_func(params, **fixed_args)
        # Optuna maximizes by default
        return metric
    return objective

def run_optuna(runner_func, fixed_args, n_trials=50, out_dir="optuna_results"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    study = optuna.create_study(direction="maximize")
    objective = objective_factory(runner_func, fixed_args)
    study.optimize(objective, n_trials=n_trials)
    best = study.best_trial.params
    with open(os.path.join(out_dir, "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)
    return study, best
