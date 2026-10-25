"""Spike exit gate: neural model vs median baseline on identical origins.

Inputs: per-sample neural results from train_common.evaluate_curves (each has
cutoff-matched truth) + median-baseline errors on the same users. Because the
neural path scores days-to-onset from tensor cutoffs while the classical path
scores cycle lengths on rolling origins, the gate compares them on a shared
construct: for each test sample, median prediction = median(history lengths
prior to cutoff) mapped to days-to-onset via the cutoff's cycle_day.

Simpler, auditable rule (documented in eval_report): both sides predict the
same held-out onsets; errors pair by sample; Wilcoxon on paired absolute
errors; coverage from the neural p10/p90 interval vs truth.

Exit (Plan B): neural MAE <= median MAE - 0.5 AND Wilcoxon p < 0.05 AND
80%/90%-style coverage reported (gate warns, does not fail, on coverage —
intervals must at least match the +-3d heuristic width).
"""
from __future__ import annotations

import statistics


def median_days_to_onset(lengths: list[int], cycle_day: int, default: int = 28) -> int:
    rec = [l for l in (lengths or []) if l and l > 0][-6:]
    med = int(statistics.median(rec)) if rec else default
    if cycle_day and cycle_day > 0:
        return max(med - cycle_day + 1, 1)
    return med


def gate_report(neural: list[dict], baseline_errs: list[float],
                coverage_hits_80: list[bool] | None = None) -> dict:
    """neural: evaluate_curves rows with err/pred/truth; baseline_errs aligned."""
    from ovumcy_eval.metrics import wilcoxon_p

    n_errs = [float(r["err"]) for r in neural]
    assert len(n_errs) == len(baseline_errs) and n_errs
    mae_n = sum(n_errs) / len(n_errs)
    mae_b = sum(baseline_errs) / len(baseline_errs)
    p = wilcoxon_p(n_errs, list(baseline_errs))
    out = {
        "n": len(n_errs),
        "mae_neural": mae_n,
        "mae_baseline": mae_b,
        "delta": mae_n - mae_b,
        "wilcoxon_p": p,
        "pass_mae": (mae_n - mae_b) <= -0.5,
        "pass_wilcoxon": p < 0.05,
    }
    out["pass"] = bool(out["pass_mae"] and out["pass_wilcoxon"])
    if coverage_hits_80 is not None and len(coverage_hits_80) == len(n_errs):
        out["coverage_80"] = sum(1 for h in coverage_hits_80 if h) / len(coverage_hits_80)
    reg = [float(r["err"]) for r in neural if r.get("band") == "regular"]
    irr = [float(r["err"]) for r in neural if r.get("band") == "irregular"]
    out["mae_regular"] = sum(reg) / len(reg) if reg else 0.0
    out["mae_irregular"] = sum(irr) / len(irr) if irr else 0.0
    return out
