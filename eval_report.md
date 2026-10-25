# Eval report — full run 2026-10-24 (seed 42)

> Track B gate **PASS** on the 400-user synthetic run below. The earlier
> CPU-seconds proof (`retrain.sh --smoke-train`, 1 epoch, expected FAIL) is
> kept at the bottom as plumbing history.

## How to reproduce

```bash
bash scripts/retrain.sh --check         # fast smoke, no training
bash scripts/retrain.sh --smoke-train   # CPU-seconds proof (this section)
bash scripts/retrain.sh --full          # FINAL training (hours) + release table
```

## Smoke baseline (auto-filled by --check)

See `artifacts/smoke/baseline_report.json` after running the check.
Median6 MAE 6.67 (regular 2.97 / irregular 14.06) — the bar.

## Plumbing history — proof run (1 epoch, 6 users x 4 cycles, CPU seconds)

- Spike B PatchTST-tiny (1,255,616 params): neural MAE 12.00 vs median 9.88
  (delta +2.12), Wilcoxon p=0.34 → gate FAIL (expected: untrained).
  Coverage-80 0.25. Report: `artifacts/smoke/patchtst/report.json`.
- Personal (2 users x 8 cycles, rank-4 bottleneck head): ctx MAE 6.83 vs
  adapter 11.00 (delta +4.17) → winner: context (expected on 1-epoch base).
  0.03s/user, 1124B int8 (budget 2048). Report:
  `artifacts/smoke/personal/report.json`.
- Export: `artifacts/smoke/model.onnx` (opset 17, static [1,180,12]→[1,3,32],
  INT8 kept, spot MAE unchanged). Sidecar `healthz` reports `"neural": true`;
  `/v1/predict` contract check passes (censored proof model correctly falls
  back to the median heuristic).
- Spike A: not runnable in this sandbox (no IBM toolkit backend; exits
  non-zero with install instructions, trains nothing). Runnable wherever
  `pip install git+https://github.com/ibm-granite/granite-timeseries` works.

## Release backtest — 2026-10-24 full run (seed 42, synthetic 400 users x 8 cycles)

| model | MAE all | MAE regular | MAE irregular | 80% cov | Wilcoxon p | gate |
|---|---|---|---|---|---|---|
| median6 (bar) | 8.56 | — | — | — | — | — |
| Spike A TTM | — | — | — | — | — | SKIP (no backend in sandbox; see below) |
| Spike B PatchTST-tiny (1.43M, 71 epochs, 2560 train / 640 val) | 3.84 | 3.12 | 5.43 | 0.67 | 1.6e-27 | **PASS** (delta −4.72) |

- Spike B beats median6 by 4.72d (bar: −0.5d), Wilcoxon p≈1.6e-27 (bar: <0.05).
- Coverage 0.67 is below the 0.80 exit bar: intervals are too narrow (soft
  targets with w=2 under-spread the quantiles). Safe end-to-end anyway —
  Plan A widens neural windows to at least legacy width, and censored outputs
  fall back to the median heuristic. Follow-up before claiming calibration:
  widen `smooth_w` and/or temperature-scale curves on the val split.
- Personal spike (20 users, rank-4 bottleneck head on frozen base): context
  7.75 vs adapter 8.30 (delta +0.55) → winner **context**. Confirms the plan's
  prior: at n=8 (<12) adapters overfit; keep the n≥12 gate for adapters.
  0.04s/user, 1124B int8 (budget 2048). Report: `artifacts/personal/report.json`.
- Export: `artifacts/model.onnx` fp32 (INT8 spot check regressed 4.000→4.125,
  so fp32 kept per the lossless-only rule). Serve `healthz` reports
  `"neural": true`; contract check passes.

Decision: ship Spike B (Track B). Track A stays open — rerun
`train_spike_a --full` where the IBM toolkit installs, then `--from auto` to
compare; the gate picks the winner.

## Fehring real-data eval — Phase A v1 (2026-10-25, seed 42, no weights change)

Reproduce: `python scripts/eda_fehring.py --csv data/fehring_raw.csv --out artifacts/eda`
then `python scripts/eval_fehring.py --csv data/fehring_raw.csv --out artifacts/fehring/eval_v1.json`.
The CSV is licensed/local-only (gitignored) — only aggregates leave `data/`.

- Loader: `src/ovumcy_data/fehring.py` chains de-duplicated cycles
  (1649 kept / 16 dup rows dropped) into 159 timelines (48,352 days);
  flow map blank→none / 1→light / 2→medium / 3→heavy; peak rule EDO-positioned
  with FirstDayofHigh+4 fallback; BBT fully masked; reconstructed days present.
  EDA + decisions: `artifacts/eda/fehring_eda.md`.
- Samples (v1 tensor, `build_samples`, seed 42): **1139** across 159 users
  (censored 2). Median6 baseline on identical origins:
  | slice | n | MAE | MedAE |
  |---|---|---|---|
  | all | 1139 | 6.79 | 2.00 |
  | regular (sd≤3) | 836 | 5.92 | 2.00 |
  | irregular | 303 | 9.20 | 5.00 |
  | arm0 | 667 | 6.58 | 2.00 |
  | arm1 | 472 | 7.10 | 2.00 |
  | typical bleeding | 1073 | 6.58 | 2.00 |
  | unusual (flag) | 66 | 10.35 | 6.00 |
  | cycling | 1065 | 6.57 | 2.00 |
  | postpartum (flag) | 74 | 9.97 | 6.00 |
- Neural on Fehring (v1 checkpoint `artifacts/patchtst/model.pt`, filled
  2026-10-25 now that torch installs): MAE **5.08** vs median 6.79
  (delta −1.72), Wilcoxon p=0.14, coverage-80 0.53, regular 4.97 /
  irregular 5.39 → gate **FAIL** (p ≥ 0.05). Real-data neural is 1.24d worse
  than synthetic (3.84) and the peak column stays unvalidated — the retrain
  was justified. Full report: `artifacts/fehring/eval_v1.json`.
- Gate to Phase B: **PROCEED** (ran as prep, then as the full retrain below).

## Phase B mixed retrain — 2026-10-25 (seed 42, tensor v2, from scratch)

Reproduce: `bash scripts/retrain.sh --full-v2`
(`train_mixed_v2 --full` → `calibrate_v2` → `export_onnx --from mixed_v2 --tensor v2`).
Pool: 400 synthetic users × 8 cycles + 159 Fehring timelines (1649/1665 rows
kept; postpartum cutoffs dropped from the *train* split only), 80/20
by-client split (447 train / 112 val users; no client in both), seed 42.
Samples: v2 tensor (14 cols), 3404 train / 886 val. Model: PatchTST-tiny with
the shipped knobs, head 12→14 ch (1,459,456 params), 80 epochs CPU, patience 8
(ran all 80; best = epoch 80). Peak supervision: 23.3% of train windows and
18.0% of val windows contain a peak=1 day (Fehring EDO signal; synthetic days
are peak-unknown).

| model | MAE all | MAE regular | MAE irregular | 80% cov | Wilcoxon p | gate |
|---|---|---|---|---|---|---|
| median6 on mixed val (bar) | 9.07 | — | — | — | — | — |
| mixed-v2 PatchTST (1.46M, 80 epochs, 3404 train / 886 val) | 4.35 | 3.73 | 5.51 | 0.54 → **0.84** cal. | 1.5e-23 | **PASS** (delta −4.71) |

- By source on identical val origins — fehring (n=190): neural 3.90 vs
  median 6.74 (delta −2.84); synthetic (n=696): 4.48 vs 9.70. Real-data
  regular/irregular and arm splits are in `artifacts/mixed_v2/report.json`.
- Calibration (post-decode symmetric day-widening, p50/MAE untouched):
  k=3 days, coverage 0.54 → 0.84 (target 0.80, reached), mean width
  5.8 → 11.8d. Record: `artifacts/mixed_v2/calibration.json`; embedded as
  `interval_widen_days` in the export normalizer for the app-side agent
  (sidecar stays v1 in this plan — no serve code changes).
- Export: `artifacts/mixed_v2/model.onnx` (opset 17, static [1,180,14] →
  [1,3,32], INT8 kept — spot MAE 3.625 → 3.625, lossless per the 0.1d rule).
  ONNX↔torch parity max-abs-diff 1.7e-06; decode smoke 16/16. The shipped
  `artifacts/model.onnx` (v1 fp32) is untouched. Serve wiring for 14 cols is
  app-side follow-up; `healthz`-level verification here is ONNX load + decode.

Decision: Phase B gate **PASS** on the mixed val set. No version tag until a
human approves these numbers (plan step 8).
