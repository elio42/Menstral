"""Train baselines + rolling-origin report (fast, no torch)."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ovumcy_data.synthetic import generate_corpus
from ovumcy_eval import backtest as _bt
from ovumcy_eval.baselines import mean_predict, median_predict, naive_repeat_predict
from ovumcy_eval.metrics import mae, medae, rmse, wilcoxon_p


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default="artifacts/smoke/baseline_report.json")
    args = p.parse_args()
    n_users, cyc = (12, 4) if args.smoke else (400, 8)
    corpus = generate_corpus(n_users=n_users, cycles_per_user=cyc, seed=42)
    users = corpus["users"]
    reps = {}
    for name, fn in [("median6", median_predict), ("mean6", mean_predict),
                     ("naive", naive_repeat_predict)]:
        reps[name] = _bt.rolling_origin(users, fn, n_holdout=2)
    # Wilcoxon: median vs naive on paired user MAEs
    a = [u["mae"] for u in reps["median6"]["per_user"]]
    b = [u["mae"] for u in reps["naive"]["per_user"]]
    n = min(len(a), len(b))
    pval = wilcoxon_p(a[:n], b[:n]) if n else 1.0
    report = {
        "seed": 42, "smoke": bool(args.smoke),
        "median6_mae": reps["median6"]["mae"],
        "median6_mae_regular": reps["median6"]["mae_regular"],
        "median6_mae_irregular": reps["median6"]["mae_irregular"],
        "mean6_mae": reps["mean6"]["mae"],
        "naive_mae": reps["naive"]["mae"],
        "wilcoxon_median_vs_naive_p": pval,
        "bar": "transformer must beat median6 by -0.5d, Wilcoxon p<0.05",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
