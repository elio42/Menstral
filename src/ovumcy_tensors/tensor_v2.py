"""Daily tensor v2 — v1 (12 cols) + peak-day columns.

Frozen v2 feature order for the model matrix [180, 14]:

  0  bleed            0/1 non-spotting period day (unchanged from v1)
  1  flow_norm        flow 0-4 / 4 (unchanged)
  2  bbt_z            (unchanged; Fehring rows are always 0 with bbt_missing=1)
  3  bbt_missing      (unchanged; always 1 on Fehring rows — never synthesized)
  4  mucus_norm       (unchanged; Fehring rows always 0 — no daily mucus)
  5  symptom_log      (unchanged; Fehring rows always 0)
  6  disturbed        (unchanged; Fehring rows always 0)
  7  cycle_day_norm   (unchanged)
  8  cycle_day_missing(unchanged)
  9  gap_before       (unchanged)
  10 present          (unchanged; Fehring reconstructed days are present=1)
  11 absent           (unchanged; 1-present)
  12 peak             0/1 peak-day marker (NEW)
  13 peak_missing     0/1 peak unknown flag (NEW)

Peak semantics (from the Fehring loader rule, adopted verbatim by the app):
  peak=1 on exactly one day per ovulatory cycle (the EDO day, or the
  FirstDayofHigh+4 fallback); 0 elsewhere and on every day of anovulatory
  cycles. peak_missing=1 when the position is estimated (EDO blank fallback)
  or unknown (both EDO and FirstDayofHigh blank, a day with no log, or a
  log with no peak keys at all such as synthetic users);
  0 when the value is observed (EDO-positioned peak, or a known-absent day
  of an anovulatory cycle / non-peak day of an EDO-positioned cycle).

  The app's libido field adopts the identical column meaning later: libido
  readings NEVER set peak=1 by themselves; a confirmed peak day (same
  definition) sets peak=1 with peak_missing=0, otherwise peak=0 and
  peak_missing=1 when no peak information exists for that day.

Old v1 weights stay valid for v1; v2 changes the input dim (12 -> 14) so it
needs the Phase B retrain. The Go mirror lives in Plan A land — the exact
wire column spec is filed as docs/tensor-v2-interface.md for the app-side
agent; this module changes only the Python builder, to_model_matrix, and
parity tests. No Go changes here.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from ovumcy_tensors.tensor_v1 import (
    SKIP_GAP_DAYS,
    WINDOW_DAYS,
    build_tensor as _build_v1,
    to_model_matrix as _matrix_v1,
)

VERSION = "v2"
NUM_FEATURES = 14
FEATURE_COLUMNS = [
    "bleed",
    "flow_norm",
    "bbt_z",
    "bbt_missing",
    "mucus_norm",
    "symptom_log",
    "disturbed",
    "cycle_day_norm",
    "cycle_day_missing",
    "gap_before",
    "present",
    "absent",
    "peak",
    "peak_missing",
]


def _peak_of(log: Any) -> tuple[float, float]:
    if not isinstance(log, dict):
        return 0.0, 1.0  # absent day: unknown
    if log.get("peak_day"):
        return 1.0, 0.0 if not log.get("peak_missing") else 1.0
    # Present day, no peak marker. Explicit peak_missing=False (Fehring loader
    # on EDO-positioned / anovulatory cycles) means known-absent; a log with
    # no peak keys at all (e.g. synthetic users) carries no peak information.
    if "peak_missing" not in log:
        return 0.0, 1.0
    return 0.0, 1.0 if log.get("peak_missing") else 0.0


def build_tensor_v2(logs: list[dict], last_period_start: Any, today: Any) -> dict:
    """Build v2 tensor over [today-179, today]. Same rows as v1 + peak cols."""
    v1 = _build_v1(logs, last_period_start, today)
    by_day: dict[str, dict] = {}
    for log in logs or []:
        from ovumcy_tensors.tensor_v1 import parse_day

        d = parse_day(log.get("date"))
        if d is None:
            continue
        by_day[d.isoformat()] = log  # latest wins, same as v1
    days = []
    for row in v1["days"]:
        if row.get("present"):
            peak, peak_m = _peak_of(by_day.get(row["date"]))
        else:
            peak, peak_m = 0.0, 1.0
        r = dict(row)
        r["peak"] = int(peak)
        r["peak_missing"] = int(peak_m)
        days.append(r)
    return {"version": VERSION, "days": days}


def to_model_matrix_v2(
    tensor: dict,
    bbt_mean: float = 36.80,
    bbt_std: float = 0.35,
    cycle_day_scale: float = 35.0,
) -> "list[list[float]]":
    """Wire rows -> z-normalized float matrix [180, 14]. No torch needed."""
    base = _matrix_v1(tensor, bbt_mean, bbt_std, cycle_day_scale)
    by_date = {r.get("date"): r for r in tensor.get("days", [])}
    # base rows follow tensor["days"] order, so zip is aligned; look up peak
    # from the same ordered rows to avoid date-key fragility.
    out: list[list[float]] = []
    for row_m, row in zip(base, tensor.get("days", [])):
        out.append(list(row_m) + [float(row.get("peak", 0)), float(row.get("peak_missing", 1))])
    assert all(len(r) == NUM_FEATURES for r in out), "v2 matrix must be [180, 14]"
    return out
