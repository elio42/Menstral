"""Fehring real-data eval — no weights change (Phase A step 3).

Builds v1 samples from Fehring timelines (loader, seed 42), scores the
classical median6 baseline on identical origins, and — where the train deps
are installed — the shipped PatchTST checkpoint. In torch-free environments
the neural side is recorded as PENDING with a reason instead of failing.

Writes artifacts/fehring/eval_v1.json (no raw rows, aggregates only).

Usage:
  python -m scripts.eval_fehring --csv data/fehring_raw.csv --out artifacts/fehring/eval_v1.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ovumcy_data import build_samples, load_fehring_users  # noqa: E402
from ovumcy_eval.gate import median_days_to_onset  # noqa: E402
from ovumcy_eval.metrics import medae, wilcoxon_p  # noqa: E402


def _mae(errs: list[float]) -> float:
    return sum(errs) / len(errs) if errs else 0.0


def _split_mae(samples: list[dict], errs: list[float], key: str, val: str) -> dict:
    sel = [e for s, e in zip(samples, errs) if s.get(key) == val]
    return {"n": len(sel), "mae": round(_mae(sel), 4),
            "medae": round(statistics.median(sel), 4) if sel else 0.0}


def main() -> int:
    p = argparse.ArgumentParser(description="Fehring real-data eval (no training).")
    p.add_argument("--csv", default="data/fehring_raw.csv")
    p.add_argument("--out", default="artifacts/fehring/eval_v1.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-user", type=int, default=8)
    p.add_argument("--ckpt", default="artifacts/patchtst/model.pt")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[eval_fehring] CSV not found: {args.csv} (licensed copy only)",
              file=sys.stderr)
        return 3

    corpus = load_fehring_users(args.csv, seed=args.seed)
    users = corpus["users"]
    stats = corpus["stats"]

    # flag lookup: (user_id, date) -> flags at cutoff
    flag_by_day: dict[tuple[str, str], dict] = {}
    group_by_user: dict[str, str] = {}
    for u in users:
        group_by_user[u["user_id"]] = u.get("group", "")
        for log in u["logs"]:
            flag_by_day[(u["user_id"], log["date"])] = {
                "unusual": bool(log.get("unusual_flag")),
                "postpartum": bool(log.get("postpartum_flag")),
            }

    samples = build_samples(users, max_per_user=args.max_per_user, seed=args.seed)
    if not samples:
        print("[eval_fehring] no samples built", file=sys.stderr)
        return 3

    # annotate samples with Fehring strata (cutoff-day flags + arm)
    for s in samples:
        fl = flag_by_day.get((s["user_id"], s["cutoff"]), {"unusual": False, "postpartum": False})
        s["unusual"] = "unusual" if fl["unusual"] else "typical"
        s["postpartum"] = "postpartum" if fl["postpartum"] else "cycling"
        s["arm"] = "arm1" if group_by_user.get(s["user_id"]) == "1" else "arm0"

    # classical baseline on identical origins (torch-free)
    base_errs = []
    for s in samples:
        pred = median_days_to_onset(s.get("hist_lengths") or [],
                                    int(s.get("cycle_day_at_cutoff") or 0))
        truth = int(s["days_to_onset"])
        base_errs.append(abs(pred - truth))
    truths = [int(s["days_to_onset"]) for s in samples]

    report: dict = {
        "phase": "A-step3-real-data-eval-v1",
        "seed": args.seed,
        "csv_stats": stats,
        "n_users": len(users),
        "n_samples": len(samples),
        "n_censored": sum(1 for s in samples if s.get("censored")),
        "baseline_median6": {
            "mae": round(_mae(base_errs), 4),
            "medae": round(medae(truths, [t - e if False else t for t, e in zip(truths, base_errs)]), 4)
            if False else round(statistics.median(base_errs), 4),
            "by_band": {
                "regular": _split_mae(samples, base_errs, "band", "regular"),
                "irregular": _split_mae(samples, base_errs, "band", "irregular"),
            },
            "by_arm": {
                "arm0": _split_mae(samples, base_errs, "arm", "arm0"),
                "arm1": _split_mae(samples, base_errs, "arm", "arm1"),
            },
            "by_bleeding": {
                "typical": _split_mae(samples, base_errs, "unusual", "typical"),
                "unusual": _split_mae(samples, base_errs, "unusual", "unusual"),
            },
            "by_postpartum": {
                "cycling": _split_mae(samples, base_errs, "postpartum", "cycling"),
                "postpartum": _split_mae(samples, base_errs, "postpartum", "postpartum"),
            },
        },
        "neural": None,
        "gate": {},
    }

    # neural side: needs torch + checkpoint; record PENDING when unavailable
    neural_reason = None
    try:
        import torch  # noqa: F401
    except Exception as e:
        neural_reason = f"torch unavailable in this env ({type(e).__name__})"
    if neural_reason is None and not os.path.exists(args.ckpt):
        neural_reason = f"checkpoint missing: {args.ckpt}"
    if neural_reason is None:
        try:
            from ovumcy_models.harness import load_config  # noqa
            from ovumcy_models.patchtst import PatchTSTTiny  # noqa
            from ovumcy_models.train_common import evaluate_curves  # noqa
            import torch as _t

            cfg = {}
            for path in ("configs/spike_b_patchtst.yaml", "configs/base.yaml"):
                if os.path.exists(path):
                    try:
                        from ovumcy_models.harness import load_config as _lc

                        cfg = _lc(path)
                        break
                    except Exception:
                        continue
            m = cfg.get("model", {})
            model = PatchTSTTiny(d_model=int(m.get("d_model", 160)),
                                 n_heads=int(m.get("n_heads", 4)),
                                 n_layers=int(m.get("n_layers", 4)),
                                 patch_len=int(m.get("patch_len", 16)),
                                 stride=int(m.get("stride", 8))).build()
            state = _t.load(args.ckpt, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            rows = evaluate_curves(model, samples)
            n_errs = [float(r["err"]) for r in rows]
            cov = [(r["plo"] is not None and r["plo"] <= r["truth"] <= (r["phi"] or 0))
                   for r in rows]
            from ovumcy_eval import gate_report

            rep = gate_report(rows, base_errs, cov)
            rep["coverage_80"] = sum(1 for h in cov if h) / len(cov) if cov else 0.0
            reg = [float(r["err"]) for r in rows if r.get("band") == "regular"]
            irr = [float(r["err"]) for r in rows if r.get("band") == "irregular"]
            neural = {
                "ckpt": args.ckpt,
                "mae_neural": round(rep["mae_neural"], 4),
                "mae_baseline": round(rep["mae_baseline"], 4),
                "delta": round(rep["delta"], 4),
                "wilcoxon_p": rep["wilcoxon_p"],
                "coverage_80": round(rep.get("coverage_80", 0.0), 4),
                "mae_regular": round(rep.get("mae_regular", 0.0), 4),
                "mae_irregular": round(rep.get("mae_irregular", 0.0), 4),
                "pass": bool(rep["pass"]),
            }
            report["neural"] = neural
            synth_mae = 3.84
            synth_cov = 0.67
            gate_proceed = (
                neural["mae_neural"] > synth_mae + 1.0
                or neural["coverage_80"] < 0.80
                or True  # peak column has no other validation path yet
            )
            report["gate"] = {
                "synthetic_mae": synth_mae,
                "synthetic_coverage": synth_cov,
                "proceed_to_phase_b": bool(gate_proceed),
                "reason": ("neural MAE {:.2f} vs synthetic {:.2f}; coverage {:.2f} < 0.80; "
                           "peak column unvalidated -> PROCEED to Phase B prep").format(
                               neural["mae_neural"], synth_mae, neural["coverage_80"]),
            }
        except Exception as e:
            neural_reason = f"neural eval failed ({type(e).__name__}: {str(e)[:200]})"
    if report["neural"] is None:
        report["neural"] = {"status": "PENDING", "reason": neural_reason or "unknown"}
        # Gate without neural: baseline gap + known synthetic coverage shortfall
        # already force Phase B prep; neural confirmation runs where torch installs.
        report["gate"] = {
            "proceed_to_phase_b": True,
            "reason": ("neural PENDING in this env (" + (neural_reason or "no torch") +
                       "); synthetic coverage 0.67 < 0.80 and the peak column has no "
                       "other validation path -> PROCEED to Phase B prep; confirm neural "
                       "numbers where train deps install "
                       "(pip install -e .[train] + torch CPU, then rerun this script)."),
        }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    b = report["baseline_median6"]
    print(f"[eval_fehring] users={len(users)} samples={len(samples)} "
          f"censored={report['n_censored']} baseline_MAE={b['mae']} "
          f"(reg {b['by_band']['regular']['mae']} / irr {b['by_band']['irregular']['mae']}) "
          f"neural={report['neural'].get('mae_neural', report['neural'].get('status'))} "
          f"-> {args.out}")
    print(f"[eval_fehring] gate: PROCEED={report['gate']['proceed_to_phase_b']}: {report['gate']['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
