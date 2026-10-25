"""Phase B step 7 — calibration pass: widen intervals to coverage 0.80 (small).

Method (auditable, p50/MAE untouched): post-decode symmetric day-widening.
For each val sample with a non-censored prediction, interval [plo, phi]
becomes [plo-k, phi+k]; censored rows stay censored. Sweep integer k = 0..8
on the frozen val split, pick the smallest k with coverage >= 0.80. If no k
reaches it, keep the best k and say so (gate reports coverage; the exit bar
is MAE delta + Wilcoxon, coverage reported).

The widening is a serving-time display parameter, not a weight change:
it is recorded here and embedded into the export normalizer.json for the
app-side agent. No serve code changes in this plan (sidecar stays v1).

Usage:
  python -m scripts.calibrate_v2 --ckpt artifacts/mixed_v2/model.pt \
      --out artifacts/mixed_v2/calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _coverage(rows: list[dict], k: int) -> float:
    hit = 0
    for r in rows:
        if r.get("pred") is None or r.get("plo") is None or r.get("phi") is None:
            continue
        if r["plo"] - k <= r["truth"] <= r["phi"] + k:
            hit += 1
    n = sum(1 for r in rows if r.get("pred") is not None)
    return hit / n if n else 0.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mixed_v2_patchtst.yaml")
    p.add_argument("--ckpt", default="artifacts/mixed_v2/model.pt")
    p.add_argument("--out", default="artifacts/mixed_v2/calibration.json")
    p.add_argument("--target", type=float, default=0.80)
    p.add_argument("--kmax", type=int, default=8)
    args = p.parse_args()

    try:
        import torch  # noqa: F401
    except Exception as e:
        raise SystemExit(f"[calibrate] missing train deps: {e}")
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"[calibrate] checkpoint missing: {args.ckpt}")

    from ovumcy_data import build_samples
    from ovumcy_data.mixed import load_mixed_users
    from ovumcy_models.harness import load_config, seed_all, split_users
    from ovumcy_models.patchtst import PatchTSTTiny
    from ovumcy_models.train_common import evaluate_curves

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    data_cfg = cfg.get("data", {})
    seed = int(cfg.get("seed", 42))
    seed_all(seed)

    mix = load_mixed_users(data_cfg.get("fehring_csv", "data/fehring_raw.csv"),
                           n_synthetic_users=int(data_cfg.get("full_users", 400)),
                           cycles_per_user=int(data_cfg.get("full_cycles_per_user", 8)),
                           seed=seed)
    _, val_users = split_users(mix["users"], seed=seed)
    val_samples = build_samples(val_users, seed=seed + 1, tensor_version="v2")

    m = cfg.get("model", {})
    model = PatchTSTTiny(d_model=int(m.get("d_model", 160)),
                         n_heads=int(m.get("n_heads", 4)),
                         n_layers=int(m.get("n_layers", 4)),
                         patch_len=int(m.get("patch_len", 16)),
                         stride=int(m.get("stride", 8)),
                         dropout=float(m.get("dropout", 0.1)),
                         n_channels=int(m.get("n_channels", 14))).build()
    import torch as _t
    model.load_state_dict(_t.load(args.ckpt, map_location="cpu", weights_only=True))
    model.eval()

    rows = evaluate_curves(model, val_samples)
    mae = sum(r["err"] for r in rows) / len(rows)
    cov0 = _coverage(rows, 0)
    best_k, best_cov = 0, cov0
    for k in range(0, args.kmax + 1):
        c = _coverage(rows, k)
        if c > best_cov:
            best_k, best_cov = k, c
        if c >= args.target:
            best_k, best_cov = k, c
            break
    widths = [(r["phi"] - r["plo"]) if r.get("phi") is not None else None for r in rows]
    widths = [w for w in widths if w is not None]
    report = {
        "method": "post-decode symmetric day-widening [plo-k, phi+k], p50 untouched",
        "ckpt": args.ckpt,
        "seed": seed,
        "n_val": len(rows),
        "n_predicted": sum(1 for r in rows if r.get("pred") is not None),
        "n_censored": sum(1 for r in rows if r.get("pred") is None),
        "mae_neural": round(mae, 4),
        "coverage_before": round(cov0, 4),
        "interval_widen_days": best_k,
        "coverage_after": round(best_cov, 4),
        "target": args.target,
        "reached": bool(best_cov >= args.target),
        "mean_width_before": round(statistics.mean(widths), 3) if widths else 0.0,
        "mean_width_after": round(statistics.mean(widths) + 2 * best_k, 3) if widths else 0.0,
        "note": ("Serve integration is app-side (sidecar stays v1 in this plan); "
                 "the widening ships as normalizer.json interval_widen_days."),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[calibrate] k={best_k} coverage {cov0:.3f} -> {best_cov:.3f} "
          f"(target {args.target}, mae {mae:.3f}d unchanged) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
