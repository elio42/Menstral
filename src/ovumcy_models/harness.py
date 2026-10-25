"""Shared spike harness: config, seeding, data splits, train loop, reporting.

Both spikes share everything except build_model(). Torch is imported lazily
inside functions so --dry-run and docs builds never need it.
"""
from __future__ import annotations

import json
import os
import random


def load_config(path: str) -> dict:
    """Load yaml with optional `_base_` merge (spike configs extend base.yaml)."""
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base:
        base_path = os.path.join(os.path.dirname(path), base)
        with open(base_path) as f:
            merged = yaml.safe_load(f) or {}
        _deep_update(merged, cfg)
        return merged
    return cfg


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def seed_all(seed: int = 42):
    rng = random.Random(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    return rng


def split_users(users: list[dict], seed: int = 42, val_frac: float = 0.2):
    """Deterministic by-user train/val split (no user in both)."""
    rng = random.Random(seed + 999)
    idx = list(range(len(users)))
    rng.shuffle(idx)
    n_val = max(1, int(round(len(users) * val_frac)))
    val_ids = set(idx[:n_val])
    return [u for i, u in enumerate(users) if i not in val_ids], \
           [u for i, u in enumerate(users) if i in val_ids]


def run_training(tag: str, cfg: dict, build_model, out_dir: str,
                epochs: int, n_users: int, cycles_per_user: int,
                device: str = "cpu", patience: int | None = 8,
                tsmix_p: float = 0.3) -> dict:
    """Full train + validate + gate loop. Returns the JSON-serializable report."""
    import torch

    from ovumcy_data import build_samples, generate_corpus
    from ovumcy_eval import gate_report
    from ovumcy_models.train_common import evaluate_curves, train_one_epoch

    seed = int(cfg.get("seed", 42))
    rng = seed_all(seed)
    batch_size = int(cfg.get("training", {}).get("batch_size", 32))
    lr = float(cfg.get("learning_rate", 0.0005))

    corpus = generate_corpus(n_users=n_users, cycles_per_user=cycles_per_user,
                             seed=int(cfg.get("data", {}).get("synthetic_seed", seed)))
    train_users, val_users = split_users(corpus["users"], seed=seed)
    train_samples = build_samples(train_users, seed=seed)
    val_samples = build_samples(val_users, seed=seed + 1)
    if not train_samples or not val_samples:
        raise SystemExit(f"[{tag}] no samples built (users={n_users})")

    model = build_model(cfg)
    device_t = torch.device(device)
    model.to(device_t)
    n_params = sum(p.numel() for p in model.parameters())
    # weight_decay=0: on this small-data spike the default 0.01 decay shrank the
    # small-init patch/attention weights faster than gradients grew them, helping
    # the channel-collapse observed 2026-10-24. Revisit for the release run.
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.5, patience=3)

    os.makedirs(out_dir, exist_ok=True)
    best_mae, best_state, bad = float("inf"), None, 0
    history = []
    for ep in range(1, epochs + 1):
        tr = train_one_epoch(model, train_samples, optim, device, batch_size,
                             rng, augment=True, tsmix_p=tsmix_p)
        rows = evaluate_curves(model, val_samples, device)
        mae = sum(r["err"] for r in rows) / len(rows)
        sched.step(mae)
        history.append({"epoch": ep, "train_loss": tr, "val_mae": mae})
        print(f"[{tag}] epoch {ep}/{epochs} train_loss={tr:.4f} val_mae={mae:.3f}d", flush=True)
        if mae < best_mae - 1e-6:
            best_mae, bad = mae, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if patience is not None and bad >= patience:
                print(f"[{tag}] early stop at epoch {ep} (patience {patience})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    rows = evaluate_curves(model, val_samples, device)
    cov = [ (r["plo"] is not None and r["plo"] <= r["truth"] <= (r["phi"] or 0))
            for r in rows ]
    rep = gate_report(rows, [r["baseline_err"] for r in rows], cov)
    rep.update({
        "tag": tag, "seed": seed, "device": device,
        "n_train_samples": len(train_samples), "n_val_samples": len(val_samples),
        "n_params": n_params, "best_val_mae": best_mae,
        "history": history,
        "coverage_80": sum(1 for h in cov if h) / len(cov) if cov else 0.0,
    })
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[{tag}] gate: neural MAE {rep['mae_neural']:.3f} vs median "
          f"{rep['mae_baseline']:.3f} (delta {rep['delta']:+.3f}), "
          f"p={rep['wilcoxon_p']:.4f} -> {'PASS' if rep['pass'] else 'FAIL'}")
    return rep
