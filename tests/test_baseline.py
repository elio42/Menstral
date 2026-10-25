"""Baseline smoke: median runs on tiny synthetic corpus, bar documented."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ovumcy_data.synthetic import generate_corpus
from ovumcy_eval import backtest as _bt
from ovumcy_eval.baselines import median_predict


def test_baseline_smoke_runs():
    corpus = generate_corpus(n_users=6, cycles_per_user=4, seed=42)
    rep = _bt.rolling_origin(corpus["users"], median_predict, n_holdout=1)
    assert rep["n_points"] > 0 and rep["mae"] >= 0
    assert rep["mae_regular"] >= 0 and rep["mae_irregular"] >= 0
