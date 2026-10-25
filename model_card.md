# Model card — ovumcy-base-v1 (shipped) + ovumcy-mixed-v2 (candidate, 2026-10-25)

> Status: v1 **shipped, synthetic-only** (`artifacts/model.onnx` fp32, sidecar
> verified). v2 **candidate, mixed-trained, awaiting human approval** —
> `artifacts/mixed_v2/model.onnx` (INT8, lossless spot check) + calibration
> (k=3, coverage 0.84). No version tag until numbers are approved. No app/Go
> changes in this plan; sidecar stays v1, 14-col wiring is app-side follow-up.

## Intended use

Daily-tensor (v1, 180d) → next onset + ovulation + fertile window with
p10/p50/p90. Consumed by `ovumcy-web` Plan A via `POST /v1/predict` on the
shared Docker bridge (bearer token, no TLS). Transformer-only product path;
app keeps median fallback.

## Training data (AGPL-safe only)

- MIT synthetic (`cilab-ufersa`, `MenstruatioNN`, Urteaga) + CC0 Kaggle synthetic.
- Fehring/Marquette 1665 cycles BBT+mucus: generator + manifest, never raw CSV.
- Eval-only, never in weights: mcPHASES Restricted, Sympto/Clue closed.
- Forbidden: CC BY-NC/ND, TimeGPT closed, Moirai NC, TimesFM-3.0 NC.
- Seed pinned: 42. Manifest: `data_manifest.json`.

## Architecture (winner: Track B)

- Track B (shipped): PatchTST-tiny, 1,428,096 params. Channel-independent
  patch embed (16/8 over 180d), 4 layers × d=160 × 4 heads, per-channel
  patch-mean + concat head → [3,32] onset-CDF quantiles (p10/p50/p90).
  Augmentation: jitter 0.05, suffix-mixup p=0.3, day-mask 0.1. 71 epochs CPU,
  early-stopped (patience 8), 2560 train / 640 val samples, seed 42.
  Input [1,180,12] → output [1,3,32]; first crossing ≥0.5 decodes onset.
- Track A (open): `ibm-granite/granite-timeseries-ttm-r2` (Apache-2.0, public;
  the `-small` id is gated) finetuned daily `512→32` (tensors left-padded).
  Needs IBM's `granite-timeseries` toolkit — not installable in the training
  sandbox, so `artifacts/ttm/report.json` records `base_reached:false`.
  Rerun `train_spike_a --full` + `export --from auto` where it installs.
- Exit: rolling-origin MAE + 80/90% coverage + fertile hit + CPU/RAM + ONNX friction.
- Personal: frozen `n<6` → context+covariates `n=3–12` → tiny adapter r=2–4
  nightly CPU `n≥12`. Spike (n=8): context wins 7.75 vs 8.30 — gate holds.
  Never full-model copy per user. No federated learning.

## Risks / limitations

- Domain gap (no cycle/BBT in base pretraining) → stage-1 finetune required, no zero-shot deploy.
- Sparse BBT / subjective mucus → wide intervals, never narrow false precision.
- BBT-confirmed ovulation is retrospective (Zhu lag 2.7±1.9d).
- Small-n overfit (<12 cycles + adapter) → density gate, else context-only.
- Fehring Phase A (2026-10-25): median6 on real timelines is 6.79 overall
  (regular 5.92 / irregular 9.20; unusual-flag 10.35, postpartum-flag 9.97);
  v1 neural on Fehring is 5.08 vs 6.79 (delta −1.72, p=0.14 → gate FAIL —
  real data degrades vs synthetic 3.84, peak unvalidated). Full split in
  `eval_report.md` + `artifacts/fehring/eval_v1.json`.
- Mixed v2 (2026-10-25, candidate): trained from scratch on synthetic +
  Fehring (postpartum cycles excluded from train; 3404 train / 886 val
  samples, seed 42). Val MAE 4.35 vs 9.07 (delta −4.71, p≈1.5e-23,
  regular 3.73 / irregular 5.51) — **gate PASS**; fehring-source val 3.90
  vs 6.74 (delta −2.84). Coverage 0.54 raw → 0.84 after k=3 day-widening
  (p50/MAE untouched). Report: `artifacts/mixed_v2/report.json`.

## Export

`opset 17, static [1,180,C] -> [1,3,32] (C=12 v1, C=14 v2), INT8 where lossless`.
Quantile head: p10/p50/p90. `model_version` header on every reply.
v1: fp32 (INT8 regressed 4.000→4.125, kept fp32). v2 candidate: INT8 kept
(spot 3.625→3.625), parity 1.7e-06, decode smoke 16/16; normalizer carries
`interval_widen_days: 3`.

## Eval (see eval_report.md)

- v1 synthetic: MAE 3.84 vs median 8.56 (delta −4.72, bar −0.5d),
  Wilcoxon p≈1.6e-27 (bar <0.05), regular 3.12 / irregular 5.43. **Gate PASS.**
- v1 on Fehring: MAE 5.08 vs 6.79 (delta −1.72, p=0.14). **Gate FAIL** —
  motivated the mixed retrain.
- Mixed v2 on mixed val: MAE 4.35 vs 9.07 (delta −4.71, p≈1.5e-23),
  regular 3.73 / irregular 5.51, coverage 0.84 after calibration. **Gate PASS.**
- Bar: `MAE -0.5d vs median, Wilcoxon p<0.05`, stratified regular/irregular.
