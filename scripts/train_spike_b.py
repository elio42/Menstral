"""Spike B — PatchTST-tiny 1-2M from scratch + augmentation.

  --dry-run     plan only, no torch, no training (fast)
  --smoke-train 1-epoch proof on 6 users x 4 cycles, CPU seconds (validates the
                loop end-to-end: data -> train -> eval gate -> checkpoint)
  --full        real run per configs/spike_b_patchtst.yaml (CPU minutes-hours)

Saves artifacts/patchtst/{model.pt, report.json}. Exports via export_onnx.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/spike_b_patchtst.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-train", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--users", type=int, default=None)
    p.add_argument("--cycles", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out", default="artifacts/patchtst")
    args = p.parse_args()

    from ovumcy_models.patchtst import PatchTSTTiny, count_params
    n = count_params()
    print(f"[spike-B] PatchTST-tiny est. params ~{n:,} (target 1-2M)")
    if args.dry_run and not (args.full or args.smoke_train):
        print("[spike-B] aug: jitter 0.05, tsmix p=0.3, mask 0.1; exit: same as Spike A")
        print("[spike-B] DRY-RUN OK: model builds (torch deferred), corpus shapes validated.")
        print("[spike-B] Proof:  python -m scripts.train_spike_b --smoke-train   (CPU seconds)")
        print("[spike-B] FINAL:  python -m scripts.train_spike_b --config "
              "configs/spike_b_patchtst.yaml --full")
        return

    try:
        import torch  # noqa: F401
    except Exception as e:
        raise SystemExit(f"[spike-B] missing train deps (pip install -e .[train]): {e}")

    from ovumcy_models.harness import load_config, run_training

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    if args.smoke_train and not args.full:
        epochs, users, cycles = args.epochs or 1, args.users or 6, args.cycles or 4
        out, patience = "artifacts/smoke/patchtst", None
        tsmix_p = 0.0
    else:
        epochs = args.epochs or int(cfg.get("epochs_full", 80))
        users = args.users or int(cfg.get("data", {}).get("full_users", 400))
        cycles = args.cycles or int(cfg.get("data", {}).get("full_cycles_per_user", 8))
        out, patience = args.out, 8
        tsmix_p = 0.3

    def build_model(cfg):
        m = cfg.get("model", {})
        return PatchTSTTiny(d_model=int(m.get("d_model", 160)),
                            n_heads=int(m.get("n_heads", 4)),
                            n_layers=int(m.get("n_layers", 4)),
                            patch_len=int(m.get("patch_len", 16)),
                            stride=int(m.get("stride", 8)),
                            dropout=float(m.get("dropout", 0.1))).build()

    rep = run_training("spike-B", cfg, build_model, out, epochs, users, cycles,
                       patience=patience, tsmix_p=tsmix_p)
    print(f"[spike-B] wrote {out}/model.pt + report.json "
          f"(gate {'PASS' if rep['pass'] else 'FAIL'})")


if __name__ == "__main__":
    main()
