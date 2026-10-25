# Tensor v2 interface note — for the app-side (Plan A / Go) agent

> Python source of truth: `Menstral/src/ovumcy_tensors/tensor_v2.py`.
> This note changes nothing in Go; it files the exact column spec so the
> Go mirror (`BuildPredictorTensor`, `docs/predictor-contract.md` in
> `ovumcy-web`) can adopt it when Plan A schedules the work.
> No Go changes, no form changes, no API changes in the Fehring plan.

## Status

- v1 (12 cols, `[1,180,12]`) is the shipped/serve contract. Unchanged.
- v2 (14 cols, `[1,180,14]`) is implemented in Python only, proven by
  `Menstral/tests/test_tensor_v2.py` (first 12 cols byte-identical to v1).
  It needs the Phase B retrain (input dim 12 → 14) before anything serves it.
- ONNX export stays `[1,180,12] → [1,3,32]` until the v2 checkpoint passes
  the gate and a human approves the release. Do not cut a v2 export yet.

## Frozen v2 column order ([180, 14])

| idx | name | type | meaning |
|---|---|---|---|
| 0 | bleed | 0/1 | non-spotting period day (v1 unchanged) |
| 1 | flow_norm | 0–1 | flow 0–4 / 4 (v1 unchanged) |
| 2 | bbt_z | float | (v1 unchanged; Fehring rows 0 + missing) |
| 3 | bbt_missing | 0/1 | (v1 unchanged; Fehring rows always 1) |
| 4 | mucus_norm | 0–1 | (v1 unchanged; Fehring rows always 0) |
| 5 | symptom_log | 0–1 | (v1 unchanged; Fehring rows always 0) |
| 6 | disturbed | 0/1 | (v1 unchanged; Fehring rows always 0) |
| 7 | cycle_day_norm | 0–1 | cycle_day / 35 (v1 unchanged) |
| 8 | cycle_day_missing | 0/1 | (v1 unchanged) |
| 9 | gap_before | 0/1 | >90d silence (v1 unchanged) |
| 10 | present | 0/1 | (v1 unchanged; Fehring reconstructed days are 1) |
| 11 | absent | 0/1 | 1 − present (v1 unchanged) |
| 12 | **peak** | 0/1 | **NEW: peak-day marker (see semantics)** |
| 13 | **peak_missing** | 0/1 | **NEW: 1 = peak value unknown for this day** |

## peak semantics (adopt verbatim when the Go mirror lands)

- `peak=1` on exactly one day per ovulatory cycle (the EDO day, or the
  `FirstDayofHigh + 4` fallback when EDO is blank); `0` elsewhere and on
  every day of anovulatory cycles.
- `peak_missing=0` when the value is observed (EDO-positioned peak, or a
  known-absent day of an EDO-positioned / anovulatory cycle).
- `peak_missing=1` when the position is estimated (fallback cycle), unknown
  (both EDO and FirstDayofHigh blank), or the day has no log (absent row).
- The app's future libido field adopts the identical meaning: libido
  readings NEVER set `peak=1` by themselves. A confirmed peak day (same
  definition as above) sets `peak=1, peak_missing=0`; otherwise `peak=0`
  with `peak_missing=1` when no peak information exists for that day.
- Wire JSON adds `"peak"` / `"peak_missing"` next to the existing day keys
  (`src/ovumcy_tensors/tensor_v2.py::build_tensor_v2`).

## What the Go side will need (later, not this plan)

1. Extend the day row with `peak` / `peak_missing` per the table above.
2. Extend the matrix builder to 14 columns in this exact order.
3. Add golden-vector parity tests mirroring `test_tensor_v2.py`
   (`first-12-identical-to-v1`, peak-day, absent-missing cases).
4. Keep serving v1 until the v2 checkpoint + export pass the gate.
