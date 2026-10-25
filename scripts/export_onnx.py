"""Export ONNX (opset 17, static [1,180,C] -> [1,3,32]; C=12 v1, C=14 v2).

  --smoke          tiny stub to artifacts/smoke/model.onnx (no weights; validates
                   sidecar + docker paths without training)
  --full           export a spike checkpoint:
                     --from patchtst (default) | mixed_v2 | ttm | auto
                     --tensor v1 (default) | v2  (must match the checkpoint)
                     --ckpt      explicit model.pt (default: <track>/model.pt)
                     --out       default artifacts/model.onnx (v1) or
                                 artifacts/mixed_v2/model.onnx (v2)
                   INT8 dynamic quantization is applied only when a spot check
                   on matching-tensor samples shows no MAE regression (>0.1d
                   keeps fp32).

The sidecar loads --out with onnxruntime CPU and decodes curves with the same
ovumcy_serve.decode used by the eval gate, so gate numbers describe serving.
NOTE: the sidecar stays v1-pinned (12 cols) in this plan; the v2 export is
verified at the ONNX/parity level and its wiring is app-side follow-up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _pick_track(want: str) -> str:
    if want in ("patchtst", "mixed_v2", "ttm"):
        return want
    # auto: prefer a PASS report, else the fresher checkpoint.
    import glob

    tracks = {"patchtst": ("artifacts/patchtst/report.json",
                            "artifacts/patchtst/model.pt"),
              "mixed_v2": ("artifacts/mixed_v2/report.json",
                            "artifacts/mixed_v2/model.pt"),
              "ttm": ("artifacts/ttm/report.json", "artifacts/ttm/model.pt")}
    cands = []
    for track, (rep, ckpt) in tracks.items():
        if os.path.exists(ckpt):
            passed, mtime = False, 0.0
            if os.path.exists(rep):
                try:
                    r = json.load(open(rep))
                    passed = bool(r.get("pass"))
                except Exception:
                    pass
                mtime = os.path.getmtime(rep)
            cands.append((passed, mtime, track))
    if not cands:
        raise SystemExit("[export] no checkpoints found; train a spike first "
                         "(--smoke-train for the CPU-seconds proof).")
    cands.sort(reverse=True)
    return cands[0][2]


def _build_patchtst(n_channels: int = 12):
    from ovumcy_models.harness import load_config
    from ovumcy_models.patchtst import PatchTSTTiny

    cfg, m = {}, {}
    for path in ("configs/mixed_v2_patchtst.yaml",
                 "configs/spike_b_patchtst.yaml", "configs/base.yaml"):
        if os.path.exists(path):
            try:
                cfg = load_config(path)
                m = cfg.get("model", {})
                break
            except Exception:
                continue
    return PatchTSTTiny(d_model=int(m.get("d_model", 160)),
                        n_heads=int(m.get("n_heads", 4)),
                        n_layers=int(m.get("n_layers", 4)),
                        n_channels=n_channels).build()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--from", dest="source", default="auto",
                   choices=["auto", "patchtst", "mixed_v2", "ttm"])
    p.add_argument("--tensor", default="v1", choices=["v1", "v2"])
    p.add_argument("--n-channels", type=int, default=None)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    n_channels = args.n_channels or (14 if args.tensor == "v2" else 12)
    if args.out is None:
        args.out = ("artifacts/mixed_v2/model.onnx" if n_channels == 14
                    else "artifacts/model.onnx")
    if n_channels == 14 and args.out == "artifacts/model.onnx":
        # Never overwrite the shipped v1 export with v2 weights.
        args.out = "artifacts/mixed_v2/model.onnx"
        print("[export] v2 weights: redirecting out -> artifacts/mixed_v2/model.onnx "
              "(shipped v1 export stays untouched)")

    if args.smoke and not args.full:
        os.makedirs(os.path.dirname(args.out) or "." if args.out != "artifacts/smoke/model.onnx"
                    else "artifacts/smoke", exist_ok=True)
        out = "artifacts/smoke/model.onnx"
        try:
            import numpy as np
            import onnx
            from onnx import helper, TensorProto
            X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 180, 12])
            Q = helper.make_tensor_value_info("quantiles", TensorProto.FLOAT, [1, 3, 32])
            W = helper.make_tensor("W", TensorProto.FLOAT, [12, 96],
                                    np.zeros((12, 96), dtype=np.float32).ravel().tolist())
            B = helper.make_tensor("B", TensorProto.FLOAT, [96],
                                    np.zeros((96,), dtype=np.float32).ravel().tolist())
            mm = helper.make_node("Gemm", ["input_flat", "W", "B"], ["q_flat"], alpha=0.0, beta=1.0)
            reshape_in = helper.make_node("Reshape", ["input", "shape_in"], ["input_flat"])
            reshape_out = helper.make_node("Reshape", ["q_flat", "shape_out"], ["quantiles"])
            si = helper.make_tensor("shape_in", TensorProto.INT64, [2], [180, 12])
            so = helper.make_tensor("shape_out", TensorProto.INT64, [3], [1, 3, 32])
            g = helper.make_graph([reshape_in, mm, reshape_out], "smoke", [X], [Q],
                                  [W, B, si, so])
            m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
            m.ir_version = 8
            onnx.save(m, out)
            with open("artifacts/smoke/normalizer.json", "w") as f:
                json.dump({"bbt_mean": 36.8, "bbt_std": 0.35, "cycle_day_scale": 35.0,
                           "opset": 17, "smoke": True}, f, indent=2)
            print(f"[export] smoke stub -> {out} (opset 17, static [1,180,12])")
        except Exception as e:
            with open(out + ".txt", "w") as f:
                f.write(f"smoke export skipped (need pip install -e .[train]): {e}\n")
            print(f"[export] onnx not installed, marker written ({e})")
        return

    if not args.full:
        print("[export] pass --smoke (fast) or --full (real weights). Nothing done.")
        return

    try:
        import torch
    except Exception as e:
        raise SystemExit(f"[export] missing train deps (pip install -e .[train]): {e}")

    track = _pick_track(args.source)
    if track == "mixed_v2" and n_channels == 12:
        n_channels = 14
        print("[export] mixed_v2 track implies 14 channels")
    ckpt = args.ckpt or os.path.join("artifacts", track, "model.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"[export] checkpoint missing: {ckpt}")
    if track in ("patchtst", "mixed_v2"):
        model = _build_patchtst(n_channels=n_channels)
    else:
        sys.path.insert(0, "scripts")
        raise SystemExit("[export] ttm export rebuilds the backbone via "
                         "scripts.train_spike_a helpers — rerun with --from patchtst "
                         "until the TTM backend is installed (see train_spike_a --full).")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    dummy = torch.zeros(1, 180, n_channels)
    import warnings
    with warnings.catch_warnings():
        # torch 2.9+ nudges toward the torch.export-based exporter; the
        # TorchScript path is pinned here until the release validates the new one.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        torch.onnx.export(model, dummy, args.out, input_names=["input"],
                          output_names=["quantiles"], opset_version=args.opset,
                          dynamo=False)
    print(f"[export] {track} -> {args.out} (opset {args.opset}, static [1,180,{n_channels}])")

    # Spot-check fp32 vs INT8: keep fp32 on any MAE regression > 0.1d.
    # Tensor-aware: v2 checkpoints are checked on mixed v2 samples.
    try:
        import numpy as np
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_dynamic

        from ovumcy_models.train_common import evaluate_curves

        if n_channels == 14:
            from ovumcy_data import build_samples
            from ovumcy_data.mixed import load_mixed_users
            _mix = load_mixed_users("data/fehring_raw.csv", n_synthetic_users=4,
                                    cycles_per_user=4, seed=123)
            samples = build_samples(_mix["users"], seed=123,
                                    tensor_version="v2")[:8]
        else:
            from ovumcy_data import build_samples, generate_corpus
            corpus = generate_corpus(n_users=4, cycles_per_user=4, seed=123)
            samples = build_samples(corpus["users"], seed=123)[:8]
        fp_rows = evaluate_curves(model, samples)
        fp_mae = sum(r["err"] for r in fp_rows) / len(fp_rows)

        # ONNX parity: torch vs exported graph on the same samples.
        import torch as _t
        _sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        _diffs = []
        for s in samples[:4]:
            x = _t.tensor([s["X"]], dtype=_t.float32)
            with _t.no_grad():
                ref = model(x).detach().cpu().numpy()
            (got,) = _sess.run(None, {"input": np.array(s["X"], dtype=np.float32)
                                            .reshape(1, 180, n_channels)})
            _diffs.append(float(np.abs(ref[0] - got).max()))
        print(f"[export] onnx parity max-abs-diff: {max(_diffs):.2e} "
              f"(n={len(_diffs)} samples)")

        q_path = args.out + ".int8.onnx"
        quantize_dynamic(args.out, q_path, weight_type=QuantType.QInt8)

        class _Ort:
            def __init__(self, path):
                self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

            def __call__(self, x):
                import numpy as _np
                return self.s.run(None, {"input": _np.array(x.numpy()
                                         if hasattr(x, "numpy") else x,
                                         dtype=_np.float32)})

        import torch as _t

        class _Q:
            def __init__(self, path):
                self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                self.device = "cpu"

            def eval(self):
                return self

            def __call__(self, x):
                return _t.tensor(self.s.run(
                    None, {"input": x.detach().cpu().numpy().astype("float32")})[0])

        # Static batch-1 graph: run the spot check sample-by-sample.
        q_rows = evaluate_curves(_Q(q_path), samples, batch_size=1)
        q_mae = sum(r["err"] for r in q_rows) / len(q_rows)
        if q_mae - fp_mae <= 0.1:
            os.replace(q_path, args.out)
            print(f"[export] INT8 kept (spot MAE {fp_mae:.3f} -> {q_mae:.3f})")
        else:
            os.remove(q_path)
            print(f"[export] INT8 regressed ({fp_mae:.3f} -> {q_mae:.3f}); kept fp32")
    except Exception as e:
        try:
            if os.path.exists(args.out + ".int8.onnx"):
                os.remove(args.out + ".int8.onnx")
        except Exception:
            pass
        print(f"[export] quantization check skipped ({type(e).__name__}: {str(e)[:150]})")

    with open(os.path.join(os.path.dirname(args.out) or ".", "normalizer.json"), "w") as f:
        norm = {"bbt_mean": 36.8, "bbt_std": 0.35, "cycle_day_scale": 35.0,
                "opset": args.opset, "track": track, "smoke": False,
                "tensor_version": ("v2" if n_channels == 14 else "v1"),
                "n_channels": n_channels}
        # Carry the Phase B calibration (interval widening) alongside the
        # export so the app-side agent can apply it at serve time.
        calib_path = os.path.join(os.path.dirname(args.out) or ".", "calibration.json")
        if os.path.exists(calib_path):
            try:
                calib = json.load(open(calib_path))
                norm["interval_widen_days"] = calib.get("interval_widen_days", 0)
                norm["coverage_after"] = calib.get("coverage_after")
            except Exception as e:
                print(f"[export] calibration merge skipped ({e})")
        json.dump(norm, f, indent=2)
    print(f"[export] wrote {args.out} + normalizer.json — point the sidecar at it "
          f"(see docker-compose.sidecar.yml / PREDICTOR_IMAGE build).")


if __name__ == "__main__":
    main()
