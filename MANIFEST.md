# MANIFEST — what ships in the sidecar, and under what license

This file is the Plan B shipping manifest. It is checked by
`scripts/retrain.sh --check` and must be updated on every release.

## Image (`Dockerfile.cpu`)

- Base: `python:3.11-slim` (pinned by digest in Dockerfile).
- Runtime deps only: `fastapi, uvicorn, numpy, onnxruntime, pydantic, pyyaml`.
- Model artifact: `/app/artifacts/model.onnx` (opset 17, static `[1,180,12]`)
  + `/app/artifacts/normalizer.json` + `/app/artifacts/model_card.md`.
- No training code, no corpus, no personal exports in the image.
- Runs as non-root, read-only FS, `expose 8000`, no published port (see
  `docker-compose.sidecar.yml`). Shared Docker bridge only, bearer token only.

## Licenses

- Code in this repo: MIT (same as `LICENSE`).
- Base checkpoints: Track A `ibm-granite/granite-timeseries-ttm-r2`
  (Apache-2.0, CPU-explicit, public; the `-small` id is gated and needs
  `HF_TOKEN`) via IBM's `granite-timeseries` toolkit — OR Track B
  from-scratch PatchTST-tiny (MIT, no upstream weights). The release
  `model_card.md` names the winning track.
- Model I/O (both tracks): `[1,180,12]` past matrix → `[1,3,32]`
  onset-CDF quantile curves (p10/p50/p90: P(onset by day h)), opset 17, static batch 1.
  Date decoding (`first crossing >= 0.5`) is shared between the eval gate and
  the sidecar (`src/ovumcy_serve/decode.py`), so gate numbers describe serving.
- Forbidden in weights: `CC BY-NC/ND`, TimeGPT closed, Moirai NC,
  TimesFM-3.0 NC, mcPHASES Restricted, Sympto/Clue closed. Eval-only corpora
  never contribute to shipped weights; `data_manifest.json` proves it.
- Training seed pinned: `42` in every `configs/*.yaml`. Changing it requires
  a new `model_version`.

## Corresponding Source (AGPL compatibility note)

`ovumcy-web` is AGPL; this training repo is MIT. To let anyone rebuild the
shipped `model.onnx` from source, every release archives:

1. `configs/*.yaml` (exact hyperparams + seed),
2. `data_manifest.json` (every source + sha256 + access date),
3. `scripts/retrain.sh --full` (single-command rebuild),
4. `eval_report.md` (rolling-origin numbers + Wilcoxon test).

## What is NOT in the repo

- No raw Fehring/Marquette CSV (generator + manifest only).
- No `exports/json` personal corpus (local only, git-ignored).
- No `data/` outputs (git-ignored; reproduced by `build_tensors`).
- No `artifacts/*.onnx` full weights (built by `export_onnx`; smoke stub only
  in `artifacts/smoke/`).
