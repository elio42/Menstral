#!/usr/bin/env bash
# Single-command rebuild (AGPL Corresponding Source). Pinned seed 42.
#   bash scripts/retrain.sh --check        # fast: configs + contract + smoke (no torch needed)
#   bash scripts/retrain.sh --smoke-train  # CPU-seconds proof (this section)
#   bash scripts/retrain.sh --full         # FINAL training (hours) + release table
#   bash scripts/retrain.sh --full-v2      # Phase B: mixed synthetic+Fehring v2 retrain + calibrate + export
set -euo pipefail
MODE="${1:---check}"
echo "[retrain] mode=$MODE seed=42"

# Preflight: the gate needs the dev deps in the active python.
#   python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
if ! python -c "import pytest, yaml" 2>/dev/null; then
  echo "[retrain] missing deps in '$(command -v python)': run" >&2
  echo "  python -m venv .venv && . .venv/bin/activate && pip install -e \".[dev]\"" >&2
  echo "  (training also needs: pip install torch --index-url https://download.pytorch.org/whl/cpu)" >&2
  exit 3
fi

check() {
  python -m scripts.build_tensors --config configs/base.yaml --smoke --out data/smoke
  python -m scripts.train_baseline --config configs/base.yaml --smoke --out artifacts/smoke/baseline_report.json
  python -m scripts.train_spike_a --config configs/spike_a_ttm.yaml --dry-run
  python -m scripts.train_spike_b --config configs/spike_b_patchtst.yaml --dry-run
  python -m scripts.train_mixed_v2 --config configs/mixed_v2_patchtst.yaml --dry-run
  python -m scripts.train_personal --config configs/personal.yaml --dry-run
  python -m scripts.export_onnx --config configs/base.yaml --smoke
  python -m pytest -q
  echo "[retrain] CHECK green: ready for --smoke-train (proof) or --full (heavy training, not run)."
}

if [ "$MODE" = "--check" ]; then check; exit 0; fi
if [ "$MODE" = "--smoke-train" ]; then
  check
  echo "[retrain] SMOKE-TRAIN: 1-epoch proof on CPU seconds..."
  python -m scripts.train_spike_b --smoke-train
  python -m scripts.train_mixed_v2 --smoke-train
  python -m scripts.train_personal --smoke-train --base-ckpt artifacts/smoke/patchtst/model.pt
  python -m scripts.export_onnx --config configs/base.yaml --full --from patchtst --ckpt artifacts/smoke/patchtst/model.pt --out artifacts/smoke/model.onnx
  echo "[retrain] SMOKE-TRAIN done: see artifacts/smoke/{patchtst,mixed_v2,personal}/report.json"
  echo "[retrain] To serve the proof model: PREDICTOR_TOKEN=test-token MODEL_ONNX_PATH=artifacts/smoke/model.onnx uvicorn src.ovumcy_serve.app:app --port 18000"
  exit 0
fi
if [ "$MODE" = "--full" ]; then
  check
  echo "[retrain] FULL: pooled training (TTM + PatchTST + personal)..."
  # Track A is backend-gated (IBM toolkit); --allow-no-backend records the SKIP
  # in artifacts/ttm/report.json and lets Track B carry the release.
  python -m scripts.train_spike_a --config configs/spike_a_ttm.yaml --full --allow-no-backend
  python -m scripts.train_spike_b --config configs/spike_b_patchtst.yaml --full
  python -m scripts.train_personal --config configs/personal.yaml --full
  python -m scripts.export_onnx --config configs/base.yaml --full --from auto
  echo "[retrain] done: see artifacts/, model_card.md, eval_report.md"
  exit 0
fi
if [ "$MODE" = "--full-v2" ]; then
  check
  echo "[retrain] FULL-V2: mixed synthetic+Fehring v2 retrain (Phase B)..."
  python -m scripts.train_mixed_v2 --config configs/mixed_v2_patchtst.yaml --full
  python -m scripts.calibrate_v2 --ckpt artifacts/mixed_v2/model.pt --out artifacts/mixed_v2/calibration.json
  python -m scripts.export_onnx --config configs/mixed_v2_patchtst.yaml --full --from mixed_v2 --tensor v2
  echo "[retrain] done: see artifacts/mixed_v2/, model_card.md, eval_report.md"
  exit 0
fi
echo "usage: bash scripts/retrain.sh [--check|--smoke-train|--full|--full-v2]"; exit 2
