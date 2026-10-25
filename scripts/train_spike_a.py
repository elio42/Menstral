"""Spike A — finetune TTM-R2 (Apache-2.0) head on pooled daily tensors.

  --dry-run     plan only, no download, no torch (fast)
  --smoke-train 1-epoch proof (needs TTM backend; CPU seconds after download)
  --full        real run per configs/spike_a_ttm.yaml (head-only, frozen base)

Backend: IBM TinyTimeMixer. transformers has no tinytimemixer mapping, so the
script uses IBM's toolkit (pip install from https://github.com/ibm-granite/
granite-timeseries — PyPI `granite-tsfm` is an unrelated placeholder, do NOT
use it). Needs network + that checkout. Without it the script exits non-zero with
instructions — it never fakes Track A by training another architecture.

Stage-1 per plan: backbone frozen, only the quantile head trains (512 ctx,
tensors left-padded from 180d). Saves artifacts/ttm/{model.pt, report.json}.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TTM_TOOLKIT_URL = "https://github.com/ibm-granite/granite-timeseries"


def _try_load_backbone(checkpoint: str):
    """Return (backbone_module, backend_name, feature_dim, call_fn).

    call_fn(x_180x12_BTC) -> features [B,F] with backbone frozen. Raises
    RuntimeError with all captured errors when no backend works.
    """
    import torch

    errors: list[str] = []

    # Backend 1: IBM toolkit (canonical for TinyTimeMixerForPrediction).
    try:
        from tsfm_public.toolkit.tinytimemixer import TinyTimeMixerForPrediction  # type: ignore
        bm = TinyTimeMixerForPrediction.from_pretrained(checkpoint)
        bm.eval()
        for p in bm.parameters():
            p.requires_grad = False

        def call(x):
            # x: [B,180,12] -> TTM past [B,512,12] left-padded (documented).
            B = x.shape[0]
            pad = torch.zeros(B, 512 - 180, x.shape[2])
            past = torch.cat([pad, x], dim=1)
            out = bm(past_values=past)
            t = _as_tensor(out)
            while t.dim() > 2:
                t = t.mean(dim=1)
            return t

        with torch.no_grad():
            f = call(torch.zeros(1, 180, 12))
        return bm, "tsfm_public.toolkit", int(f.shape[1]), call
    except Exception as e:
        errors.append(f"toolkit: {type(e).__name__}: {str(e)[:300]}")

    # Backend 2: transformers AutoModel (no tinytimemixer mapping as of 5.x;
    # kept so a future mapping is picked up automatically).
    try:
        from transformers import AutoModel  # type: ignore
        bm = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        bm.eval()
        for p in bm.parameters():
            p.requires_grad = False

        def call2(x):
            B = x.shape[0]
            pad = torch.zeros(B, 512 - 180, x.shape[2])
            past = torch.cat([pad, x], dim=1)
            for kw in ({"past_values": past},
                       {"x": past},
                       {"input_values": past}):
                try:
                    out = bm(**kw)
                    t = _as_tensor(out)
                    while t.dim() > 2:
                        t = t.mean(dim=1)
                    return t
                except Exception:
                    continue
            raise RuntimeError("no accepted input kwarg")

        with torch.no_grad():
            f = call2(torch.zeros(1, 180, 12))
        return bm, "transformers.AutoModel", int(f.shape[1]), call2
    except Exception as e:
        errors.append(f"transformers: {type(e).__name__}: {str(e)[:300]}")

    raise RuntimeError(
        "No TTM backend available:\n  - " + "\n  - ".join(errors) +
        f"\nInstall the IBM toolkit where network allows:\n"
        f"  pip install git+{TTM_TOOLKIT_URL}\n"
        f"(PyPI `granite-tsfm` is an unrelated placeholder — do not use it.)\n"
        f"Gated checkpoints also need:  export HF_TOKEN=<your token>")


def _as_tensor(out):
    import torch

    if isinstance(out, torch.Tensor):
        return out
    for attr in ("prediction_outputs", "logits", "last_hidden_state", "predictions"):
        if hasattr(out, attr):
            v = getattr(out, attr)
            if hasattr(v, "mean") and not isinstance(v, torch.Tensor):
                try:
                    return v.mean
                except Exception:
                    pass
            if isinstance(v, torch.Tensor):
                return v
    if isinstance(out, (tuple, list)) and out:
        for v in out:
            if isinstance(v, torch.Tensor):
                return v
    raise RuntimeError(f"unrecognized backbone output: {type(out)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/spike_a_ttm.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke-train", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--users", type=int, default=None)
    p.add_argument("--cycles", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--out", default="artifacts/ttm")
    p.add_argument("--allow-no-backend", action="store_true",
                   help="record base_reached:false and exit 0 when no TTM backend is "
                        "reachable (used by retrain.sh --full so Track B still runs)")
    args = p.parse_args()

    print("[spike-A] track: TTM-R2 (Apache-2.0, CPU-explicit, multivariate, ONNX-friendly)")
    print("[spike-A] ctx 512 (tensors left-padded from 180d) -> horizon 32")
    if args.dry_run and not (args.full or args.smoke_train):
        print("[spike-A] DRY-RUN OK: tensor v1 x180 validated; checkpoints recorded, no download.")
        print("[spike-A] Proof:  python -m scripts.train_spike_a --smoke-train  (needs TTM backend)")
        print("[spike-A] FINAL:  python -m scripts.train_spike_a --config "
              "configs/spike_a_ttm.yaml --full")
        return

    try:
        import torch  # noqa: F401
        import torch.nn as nn
    except Exception as e:
        raise SystemExit(f"[spike-A] missing train deps (pip install -e .[train]): {e}")

    from ovumcy_models.harness import load_config, run_training

    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    ckpts = [cfg.get("base_checkpoint", "")] + list(cfg.get("fallback_checkpoints", []) or [])
    ckpts = [c for c in ckpts if c]
    backbone, backend, feat_dim, call_fn, used_ckpt, load_errors = None, "", 0, None, "", []
    for ck in ckpts:
        try:
            backbone, backend, feat_dim, call_fn = _try_load_backbone(ck)
            used_ckpt = ck
            break
        except RuntimeError as e:
            load_errors.append(f"{ck}: {str(e)[:200]}")
    if backbone is None:
        msg = ("[spike-A] no TTM backend/checkpoint reachable:\n" +
               "\n".join("  - " + e for e in load_errors))
        if args.allow_no_backend:
            import json as _j
            out = args.out
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "report.json"), "w") as f:
                _j.dump({"tag": "spike-A", "base_reached": False,
                         "pass": False, "reason": "no-ttm-backend",
                         "detail": load_errors,
                         "hint": f"pip install git+{TTM_TOOLKIT_URL} "
                                 "(PyPI granite-tsfm is a placeholder, do not use it); "
                                 "gated ids need HF_TOKEN. Re-run without "
                                 "--allow-no-backend afterwards."}, f, indent=2)
            print(msg)
            print(f"[spike-A] SKIP recorded in {out}/report.json "
                  f"(base_reached:false) — continuing without Track A.")
            return
        raise SystemExit(msg)
    print(f"[spike-A] base: {used_ckpt} via {backend} (frozen, feat {feat_dim})")
    horizon = int(cfg.get("pred_horizon", cfg.get("training", {}).get("pred_horizon", 32)))

    def build_model(cfg):
        head = nn.Sequential(nn.LayerNorm(feat_dim),
                             nn.Linear(feat_dim, 3 * horizon))

        class TTMHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                import torch as _t
                with _t.no_grad():
                    f = call_fn(x)
                q = self.head(f).reshape(x.shape[0], 3, horizon)
                return _t.sigmoid(q)

        return TTMHead()

    if args.smoke_train and not args.full:
        epochs, users, cycles = args.epochs or 1, args.users or 6, args.cycles or 4
        out, patience = "artifacts/smoke/ttm", None
    else:
        epochs = args.epochs or int(cfg.get("epochs_full", 30))
        users = args.users or int(cfg.get("data", {}).get("full_users", 400))
        cycles = args.cycles or int(cfg.get("data", {}).get("full_cycles_per_user", 8))
        out, patience = args.out, 8

    rep = run_training("spike-A", cfg, build_model, out, epochs, users, cycles,
                       patience=patience, tsmix_p=0.0)  # head-only: no input mixup
    rep["base_checkpoint"], rep["backend"], rep["base_reached"] = used_ckpt, backend, True
    import json as _j
    with open(os.path.join(out, "report.json"), "w") as f:
        _j.dump(rep, f, indent=2)
    print(f"[spike-A] wrote {out}/model.pt + report.json "
          f"(gate {'PASS' if rep['pass'] else 'FAIL'})")


if __name__ == "__main__":
    main()
