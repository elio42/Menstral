"""Phase B step 6 — PatchTST-tiny from scratch on synthetic + Fehring (tensor v2).

Same config knobs as the shipped v1 run (d=160, 4 heads/layers, patch 16/8,
dropout 0.1, AdamW lr 0.0005 wd 0.0, ReduceLROnPlateau, batch 32, 80 epochs,
patience 8, jitter 0.05 / tsmix p=0.3 / mask 0.1); only the input dim changes
(12 -> 14 peak/peak_missing).

  --dry-run     plan only, no torch (fast: corpus counts, no samples)
  --smoke-train 1-epoch proof on 6 synthetic x 4 + full Fehring users (v2 path
                end-to-end: mixed corpus, 14-ch model, gate)
  --full        real run per configs/mixed_v2_patchtst.yaml (CPU minutes)

Split: 80/20 BY CLIENT on the pooled user list (seed 42, never by cycle).
Train samples drop postpartum cutoffs (Phase A decision); val keeps
everything so the gate describes deployment reality. Seed 42 throughout.

Saves artifacts/mixed_v2/{model.pt, report.json}. Exports via
scripts/export_onnx.py --tensor v2.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _peak_window_frac(samples: list[dict]) -> float:
    """Fraction of samples whose 180d v2 window contains a peak=1 day."""
    hit = 0
    for s in samples:
        X = s["X"]
        w = len(X[0])
        if w >= 14 and any(float(r[12]) > 0.5 for r in X):
            hit += 1
    return hit / len(samples) if samples else 0.0


def _split_mae(errs: list[float], mask: list[bool]) -> dict:
    sel = [e for e, m in zip(errs, mask) if m]
    return {"n": len(sel),
            "mae": round(sum(sel) / len(sel), 4) if sel else 0.0,
            "medae": round(statistics.median(sel), 4) if sel else 0.0}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/mixed_v2_patchtst.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-train", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--users", type=int, default=None)
    p.add_argument("--cycles", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out", default="artifacts/mixed_v2")
    p.add_argument("--fehring-csv", default=None)
    args = p.parse_args()

    from ovumcy_models.patchtst import PatchTSTTiny, count_params
    print(f"[mixed-v2] PatchTST-tiny est. params ~{count_params(n_channels=14):,} "
          f"(14-ch head; v1 was ~{count_params():,})")

    from ovumcy_models.harness import load_config
    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    data_cfg = cfg.get("data", {})
    fehring_csv = args.fehring_csv or data_cfg.get("fehring_csv", "data/fehring_raw.csv")
    seed = int(cfg.get("seed", 42))

    if args.dry_run and not (args.full or args.smoke_train):
        from ovumcy_data.mixed import load_mixed_users
        n_s = int(data_cfg.get("full_users", 400))
        cyc = int(data_cfg.get("full_cycles_per_user", 8))
        mix = load_mixed_users(fehring_csv, n_synthetic_users=n_s,
                               cycles_per_user=cyc, seed=seed)
        fu = mix["users"]
        n_f = mix["stats"]["n_fehring_users"]
        print(f"[mixed-v2] DRY-RUN OK: {len(fu)} pooled users "
              f"({n_s} synthetic + {n_f} fehring), 80/20 by-client split, "
              f"v2 samples (train skips postpartum cutoffs).")
        print("[mixed-v2] Proof:  python -m scripts.train_mixed_v2 --smoke-train")
        print("[mixed-v2] FINAL:  python -m scripts.train_mixed_v2 --full")
        return

    try:
        import torch  # noqa: F401
    except Exception as e:
        raise SystemExit(f"[mixed-v2] missing train deps (pip install -e .[train]): {e}")

    if not os.path.exists(fehring_csv):
        raise SystemExit(f"[mixed-v2] Fehring CSV not found: {fehring_csv} "
                         f"(licensed copy only, never committed)")

    from ovumcy_data import build_samples
    from ovumcy_data.mixed import load_mixed_users
    from ovumcy_eval import gate_report
    from ovumcy_models.harness import seed_all, split_users
    from ovumcy_models.train_common import evaluate_curves, train_one_epoch

    if args.smoke_train and not args.full:
        n_s, cyc, epochs = args.users or 6, args.cycles or 4, args.epochs or 1
        out, patience, tsmix_p = "artifacts/smoke/mixed_v2", None, 0.0
    else:
        n_s = args.users or int(data_cfg.get("full_users", 400))
        cyc = args.cycles or int(data_cfg.get("full_cycles_per_user", 8))
        epochs = args.epochs or int(cfg.get("epochs_full", 80))
        out, patience, tsmix_p = args.out, 8, 0.3

    rng = seed_all(seed)
    mix = load_mixed_users(fehring_csv, n_synthetic_users=n_s,
                           cycles_per_user=cyc, seed=seed)
    users = mix["users"]
    train_users, val_users = split_users(users, seed=seed)
    src_of = {u["user_id"]: u.get("source", "?") for u in users}
    grp_of = {u["user_id"]: u.get("group", "") for u in users
              if u.get("source") == "fehring"}

    skip_pp = bool(data_cfg.get("skip_postpartum_train", True))
    train_samples = build_samples(train_users, seed=seed, tensor_version="v2",
                                  skip_postpartum_cutoffs=skip_pp)
    val_samples = build_samples(val_users, seed=seed + 1, tensor_version="v2")
    if not train_samples or not val_samples:
        raise SystemExit(f"[mixed-v2] no samples built (users={len(users)})")
    for s in val_samples:
        s["source"] = src_of.get(s["user_id"], "?")
        s["arm"] = ("arm1" if grp_of.get(s["user_id"]) == "1" else "arm0") \
            if s["source"] == "fehring" else "synthetic"

    def build_model(cfg):
        m = cfg.get("model", {})
        return PatchTSTTiny(d_model=int(m.get("d_model", 160)),
                            n_heads=int(m.get("n_heads", 4)),
                            n_layers=int(m.get("n_layers", 4)),
                            patch_len=int(m.get("patch_len", 16)),
                            stride=int(m.get("stride", 8)),
                            dropout=float(m.get("dropout", 0.1)),
                            n_channels=int(m.get("n_channels", 14))).build()

    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    model = build_model(cfg)
    device = "cpu"
    model.to(torch.device(device))
    n_params = sum(p.numel() for p in model.parameters())
    batch_size = int(cfg.get("training", {}).get("batch_size", 32))
    lr = float(cfg.get("learning_rate", 0.0005))
    optim = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                              lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=3)

    os.makedirs(out, exist_ok=True)
    best_mae, best_state, bad = float("inf"), None, 0
    history = []
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_samples, optim, device, batch_size,
                             rng, augment=True, tsmix_p=tsmix_p)
        rows = evaluate_curves(model, val_samples, device)
        mae = sum(r["err"] for r in rows) / len(rows)
        sched.step(mae)
        history.append({"epoch": ep, "train_loss": tr, "val_mae": mae})
        print(f"[mixed-v2] epoch {ep}/{epochs} train_loss={tr:.4f} "
              f"val_mae={mae:.3f}d (n_train={len(train_samples)} "
              f"n_val={len(rows)})", flush=True)
        if mae < best_mae - 1e-6:
            best_mae, bad = mae, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if patience is not None and bad >= patience:
                print(f"[mixed-v2] early stop at epoch {ep} (patience {patience})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out, "model.pt"))

    rows = evaluate_curves(model, val_samples, device)
    cov = [(r["plo"] is not None and r["plo"] <= r["truth"] <= (r["phi"] or 0))
           for r in rows]
    rep = gate_report(rows, [r["baseline_err"] for r in rows], cov)
    n_errs = [float(r["err"]) for r in rows]
    b_errs = [float(r["baseline_err"]) for r in rows]
    is_feh = [str(s.get("user_id", "")).startswith("fehring-") for s in val_samples]
    rep.update({
        "tag": "mixed-v2", "seed": seed, "device": device,
        "tensor_version": "v2", "n_channels": 14,
        "n_train_samples": len(train_samples), "n_val_samples": len(val_samples),
        "n_train_users": len(train_users), "n_val_users": len(val_users),
        "n_params": n_params, "best_val_mae": best_mae,
        "history": history,
        "coverage_80": sum(1 for h in cov if h) / len(cov) if cov else 0.0,
        "skip_postpartum_train": skip_pp,
        "train_peak_window_frac": round(_peak_window_frac(train_samples), 4),
        "val_peak_window_frac": round(_peak_window_frac(val_samples), 4),
        "val_source_split": {
            "fehring": _split_mae(n_errs, is_feh),
            "synthetic": _split_mae(n_errs, [not m for m in is_feh]),
        },
        "val_baseline_source_split": {
            "fehring": _split_mae(b_errs, is_feh),
            "synthetic": _split_mae(b_errs, [not m for m in is_feh]),
        },
    })
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[mixed-v2] gate: neural MAE {rep['mae_neural']:.3f} vs median "
          f"{rep['mae_baseline']:.3f} (delta {rep['delta']:+.3f}), "
          f"p={rep['wilcoxon_p']:.4f} cov={rep['coverage_80']:.3f} -> "
          f"{'PASS' if rep['pass'] else 'FAIL'}")
    print(f"[mixed-v2] wrote {out}/model.pt + report.json")


if __name__ == "__main__":
    main()
