"""Baselines + metrics + rolling-origin backtest (fast, dependency-light)."""
from .backtest import rolling_origin
from .baselines import mean_predict, median_predict, naive_repeat_predict
from .gate import gate_report, median_days_to_onset
from .metrics import coverage, mae, medae, rmse, wilcoxon_p

__all__ = ["median_predict", "mean_predict", "naive_repeat_predict",
           "mae", "rmse", "medae", "coverage", "wilcoxon_p", "rolling_origin",
           "gate_report", "median_days_to_onset"]
