"""Personal layer spike: context-only vs rank-constrained adapter.

  --dry-run     gating + 2KB budget check only (no torch, fast)
  --smoke-train 2-user proof on CPU seconds (needs artifacts/patchtst/model.pt;
                falls back to a fresh encoder when absent, marked in report)
  --full        real spike per configs/personal.yaml (20 users x 8 cycles)

Method: frozen PatchTST encoder + per-user bottleneck head (d->r->96, r=4:
~1124 params, ~1.1KB int8 — the LoRA-r4-equivalent capacity the plan gates on).
Context-only arm = median_days_to_onset on the same held-out origins.
Metrics: personal MAE delta + CPU s/user + adapter bytes. Saves
artifacts/personal/report.json and picks the winner for the release policy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/personal.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-train", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--base-ckpt", default="artifacts/patchtst/model.pt")
    p.add_argument("--out", default="artifacts/personal")
    args = p.parse_args()

    from ovumcy_models.adapters import adapter_fits_budget, gate_policy, lora_adapter_bytes
    print("[personal] gates: n<6 frozen | n=3-12 context+covariates | n>=12 LoRA nightly CPU")
    print(f"[personal] LoRA-r2 int8 ~{lora_adapter_bytes(2)}B, "
          f"LoRA-r4 int8 ~{lora_adapter_bytes(4)}B / 4-bit ~{lora_adapter_bytes(4, bits=4)}B "
          f"(budget 2048: r2={'FITS' if adapter_fits_budget(2) else 'OVER'} "
          f"r4={'FITS-4bit' if adapter_fits_budget(4) else 'OVER'})")
    for n in (2, 5, 8, 15):
        print(f"[personal]   n={n:2d} -> {gate_policy(n)}")
    if args.dry_run and not (args.full or args.smoke_train):
        print("[personal] spike measures on held-out users n>=6: personal MAE delta + CPU s/user + 2KB store")
        print("[personal] DRY-RUN OK: gating + budget validated, no per-user weights written.")
        print("[personal] Proof:  python -m scripts.train_personal --smoke-train  (CPU seconds)")
        print("[personal] FINAL:  python -m scripts.train_personal --config "
              "configs/personal.yaml --full")
        return

    try:
        import torch
        import torch.nn as nn
    except Exception as e:
        raise SystemExit(f"[personal] missing train deps (pip install -e .[train]): {e}")

    from ovumcy_data import build_samples, generate_corpus
    from ovumcy_eval.gate import median_days_to_onset
    from ovumcy_models.harness import load_config, seed_all
    from ovumcy_models.patchtst import PatchTSTTiny
    from ovumcy_models.train_common import pinball_loss_torch, to_tensors
    from ovumcy_serve.decode import HORIZON

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    seed = int(cfg.get("seed", 42))
    r = int((cfg.get("spike", {}) or {}).get("lora_r", 4))
    if args.smoke_train and not args.full:
        n_users, cycles, user_epochs = 2, 8, 1
        out = "artifacts/smoke/personal"
    else:
        n_users, cycles, user_epochs = 20, 8, 5
        out = args.out
    os.makedirs(out, exist_ok=True)

    seed_all(seed)
    # Base encoder: trained checkpoint when present, else fresh (marked).
    from_ckpt = os.path.exists(args.base_ckpt)
    base = PatchTSTTiny().build()
    if from_ckpt:
        base.load_state_dict(torch.load(args.base_ckpt, map_location="cpu", weights_only=True))
    for p_ in base.enc.parameters():
        p_.requires_grad = False
    base.eval()
    d = 160

    corpus = generate_corpus(n_users=n_users, cycles_per_user=cycles, seed=seed + 5)
    rows = []
    for u in corpus["users"]:
        t0 = time.time()
        samples = build_samples([u], max_per_user=8, seed=seed)
        if len(samples) < 3:
            continue
        samples.sort(key=lambda s: s["cutoff"])
        cut = max(1, int(len(samples) * 0.7))
        tr, va = samples[:cut], samples[cut:]
        # Context-only arm on the held-out origins.
        ctx_errs = []
        for s in va:
            truth_c = HORIZON + 1 if s["censored"] else int(s["days_to_onset"])
            ctx_errs.append(abs(median_days_to_onset(s["hist_lengths"],
                                                     s["cycle_day_at_cutoff"]) - truth_c))
        # Adapter arm: per-user bottleneck head, frozen encoder.
        head = nn.Sequential(nn.Linear(d, r), nn.GELU(), nn.Linear(r, 3 * 32))
        opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
        rng = seed_all(seed + abs(hash(str(u["user_id"]))) % 10000)
        head.train()
        for _ in range(user_epochs):
            for chunk in [tr[i:i + 2] for i in range(0, len(tr), 2)]:
                opt.zero_grad()
                X, y = to_tensors(chunk, augment=True, rng=rng)
                with torch.no_grad():
                    h = base.enc(base.patch(X.transpose(1, 2).reshape(
                        X.shape[0] * 12, 1, 180)).transpose(1, 2).reshape(
                        X.shape[0], 12, -1, d).mean(dim=1)).mean(dim=1)
                q = torch.sigmoid(head(h)).reshape(X.shape[0], 3, 32)
                loss = pinball_loss_torch(q, y)
                loss.backward()
                opt.step()
        # Evaluate adapter head on held-out.
        head.eval()
        ad_errs = []
        with torch.no_grad():
            for s in va:
                X = torch.tensor([s["X"]], dtype=torch.float32)
                h = base.enc(base.patch(X.transpose(1, 2).reshape(
                    1 * 12, 1, 180)).transpose(1, 2).reshape(1, 12, -1, d).mean(dim=1)).mean(dim=1)
                q = torch.sigmoid(head(h)).reshape(3, 32).tolist()
                from ovumcy_serve.decode import decode_curves
                dec = decode_curves(q, s["cutoff"])
                truth_c = HORIZON + 1 if s["censored"] else int(s["days_to_onset"])
                ad_errs.append(0.0 if (dec is None and s["censored"])
                               else float(HORIZON) if dec is None
                               else abs(int(dec["days_to_onset_p50"]) - truth_c))
        dt = time.time() - t0
        n_bytes = d * r + r + r * 96 + 96  # int8 store: 1 byte/param
        rows.append({"user_id": u["user_id"],
                     "ctx_mae": sum(ctx_errs) / len(ctx_errs),
                     "adapter_mae": sum(ad_errs) / len(ad_errs),
                     "cpu_s": dt, "adapter_bytes_int8": n_bytes})
    rep = {
        "seed": seed, "rank": r, "base_checkpoint": from_ckpt,
        "n_users": len(rows),
        "ctx_mae": sum(x["ctx_mae"] for x in rows) / max(len(rows), 1),
        "adapter_mae": sum(x["adapter_mae"] for x in rows) / max(len(rows), 1),
        "cpu_s_per_user": sum(x["cpu_s"] for x in rows) / max(len(rows), 1),
        "adapter_bytes_int8": rows[0]["adapter_bytes_int8"] if rows else 0,
        "rows": rows,
    }
    rep["delta"] = rep["adapter_mae"] - rep["ctx_mae"]
    rep["winner"] = ("adapter" if rep["delta"] <= -0.5 and
                     rep["adapter_bytes_int8"] <= 2048 else "context")
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[personal] ctx MAE {rep['ctx_mae']:.3f} vs adapter {rep['adapter_mae']:.3f} "
          f"(delta {rep['delta']:+.3f}), {rep['cpu_s_per_user']:.1f}s/user, "
          f"{rep['adapter_bytes_int8']}B -> winner: {rep['winner']}")


if __name__ == "__main__":
    main()
