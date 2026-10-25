# Menstral — Ovumcy cycle-transformer training repo (Plan B)

Companion to the `ovumcy-web` fork (Plan A). This repo produces the sidecar
model that `POST http://predictor:8000/v1/predict` serves. Transformer-only
product path; classical median stays in the app only as fallback.

> **Status: pre-training scaffold.** All pipelines are wired to the point where
> the final training step can be triggered with one command. Heavy training has
> **not** been run yet — see [How to run the final training](#how-to-run-the-final-training-validation-gate) below.

## Layout

```
configs/                    training + serve configs (seed pinned: 42)
src/ovumcy_tensors/         daily tensor v1 builder — byte-identical to Go
src/ovumcy_data/            AGPL-safe generators (synthetic MIT, Fehring-like)
src/ovumcy_eval/            baselines, metrics, rolling-origin backtest
src/ovumcy_models/          PatchTST-tiny + personal adapters (LoRA r=2-4)
src/ovumcy_serve/           FastAPI sidecar (/v1/predict + /healthz)
scripts/                    build_tensors, train_baseline, train_spike_*,
                            train_personal, export_onnx, retrain.sh
tests/                      tensor parity + DST property tests + contract tests
Dockerfile.cpu               CPU-only sidecar image
docker-compose.sidecar.yml   shared-bridge service (expose 8000, no ports)
MANIFEST.md                  what ships, licenses, corresponding source
model_card.md                model card (filled on release)
data_manifest.json           per-run {source, url, license, sha256, access_date, seed}
eval_report.md               rolling-origin results (filled on release)
```

## Wire contract (must stay byte-identical with Plan A)

`docs/predictor-contract.md` in `ovumcy-web` is normative. Tensor v1 summary:

- 180 rows, one per calendar day ending on owner today, DST-safe stepping.
- Fields: `date, present, bleed, flow(0-4), bbt_c|null, bbt_missing,
  mucus_score(0-4), symptom_count, disturbed, cycle_day, gap_before,
  missing_mask(bit0 absent, bit1 BBT, bit2 cycle_day)`.
- `skip gap >90d → gap_before=true` (merged missed-log cycles are skips, k=2).
- Request: `{user_id, tensor{version,days}, history_n, recent_lengths}`.
- Response: `{next_onset, ovulation_date, fertile_start/end, p10/p50/p90,
  model_version}` or `{no_prediction:true, reason}`. Strict decode: unknown
  fields are refused.

Python mirror: `src/ovumcy_tensors/tensor_v1.py`. Parity tests in
`tests/test_tensor_v1.py` encode the Go golden vectors.

## Data policy (AGPL-safe only)

| Source | License | How we use it |
|---|---|---|
| `cilab-ufersa`, `MenstruatioNN` scripts + synthetic CSV | MIT | generator reference + synthetic seed |
| Urteaga simulator | MIT | synthetic prior for irregularity |
| Kaggle synthetic CC0 | CC0 | pooled pre-training |
| Fehring/Marquette 1665 cycles BBT+mucus | research OK, **do not bundle raw** | generator + manifest only, never CSV |
| mcPHASES PhysioNet Restricted, Sympto/Clue closed | restricted | eval-only, never ship weights from |
| TimeGPT closed, Moirai NC, TimesFM-3.0 NC | forbidden | never touch |

Personal corpus (`exports/json`) stays local, never uploaded. Every run writes
`data_manifest.json` with `{source, url, license, sha256, access_date, seed}`.

## How to run the final training (validation gate)

Three levels, each building on the previous. `--dry-run` never touches the
network and never writes weights. `--smoke-train` is the CPU-seconds proof
(1 epoch, synthetic only). `--full` trains for hours.

```bash
# 0. Fast checks (no torch needed, ~10s)
pip install -e ".[dev]"
pytest -q                                    # tensor parity + DST + contract + dataset + gate

# 1. Build a tiny smoke corpus (synthetic only, no network)
python -m scripts.build_tensors --config configs/base.yaml --smoke --out data/smoke

# 2. Baselines — the bar the transformer must beat (MAE -0.5d, Wilcoxon p<0.05)
python -m scripts.train_baseline --config configs/base.yaml --smoke

# 3a. Spike A — finetune TTM-R2 head, daily 512→32 (needs the IBM backend, see below)
python -m scripts.train_spike_a --config configs/spike_a_ttm.yaml --dry-run
python -m scripts.train_spike_a --config configs/spike_a_ttm.yaml --full       # ← FINAL training step A

# 3b. Spike B — PatchTST-tiny 1.2M from scratch + augmentation (runnable now)
python -m scripts.train_spike_b --config configs/spike_b_patchtst.yaml --dry-run
python -m scripts.train_spike_b --smoke-train                                  # CPU-seconds proof
python -m scripts.train_spike_b --config configs/spike_b_patchtst.yaml --full # ← FINAL training step B

# 4. Personal layer — context-only vs rank-4 adapter (n≥6)
python -m scripts.train_personal --config configs/personal.yaml --dry-run
python -m scripts.train_personal --smoke-train --base-ckpt artifacts/smoke/patchtst/model.pt
python -m scripts.train_personal --config configs/personal.yaml --full         # ← FINAL personal step

# 5. Export + serve (real weights; falls back to median heuristic when censored)
python -m scripts.export_onnx --config configs/base.yaml --full --from auto
PREDICTOR_TOKEN=test-token MODEL_VERSION=ovumcy-base-v1 \
  MODEL_ONNX_PATH=artifacts/model.onnx uvicorn src.ovumcy_serve.app:app --port 8000
curl -s localhost:8000/healthz            # expect "neural": true
python -m scripts.smoke_predict --url http://localhost:8000 --token test-token

# 6. Release
bash scripts/retrain.sh --full          # pinned seed 42, writes model_card + eval_report
```

### Track A backend (one-time setup, not needed for Track B)

`transformers` has no `tinytimemixer` mapping, and PyPI `granite-tsfm` is an
unrelated placeholder — do not install it. Finetuning needs IBM's toolkit:

```bash
pip install "git+https://github.com/ibm-granite/granite-timeseries"
export HF_TOKEN=<your token>   # only for the gated -small checkpoint id
```

Checkpoints tried in order (`configs/spike_a_ttm.yaml`): the public
`ibm-granite/granite-timeseries-ttm-r2` first, the gated `-small` second.
Without the toolkit `train_spike_a --full` exits non-zero with instructions
and trains nothing — Track B is unaffected.

## Sidecar

```bash
docker compose -f docker-compose.sidecar.yml config   # validates no published ports
docker compose -f docker-compose.sidecar.yml up --build predictor
```

Auth: shared bearer token only, no TLS on the shared bridge (matches Plan A).
`model_version` header on every reply. See `MANIFEST.md` for image contents.
