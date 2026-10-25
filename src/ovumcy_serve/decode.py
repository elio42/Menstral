"""Quantile-curve decoder — shared by serve and eval.

Model I/O contract (both tracks, ONNX-stable):
  input:  [1,180,12] float32 past matrix (FEATURE_COLUMNS order)
  output: [1,3,32]  float32 onset-CDF curves, rows = p10/p50/p90

Decoding: days_to_onset(q) = first horizon day h (1-based) with curve >= 0.5.
p50 crossing -> next_onset; p10/p90 crossings -> interval. No crossing within
32d -> censored (return None: caller widens to legacy width / abstains).

Identical function here and in eval so gate numbers describe serving behavior.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

HORIZON = 32
THRESHOLD = 0.5


def first_crossing(curve: list[float], threshold: float = THRESHOLD) -> Optional[int]:
    """First 1-based horizon day with value >= threshold, else None."""
    for h, v in enumerate(curve, start=1):
        try:
            if float(v) >= threshold:
                return h
        except (TypeError, ValueError):
            continue
    return None


def decode_curves(curves: list[list[float]], cutoff: Any,
                  threshold: float = THRESHOLD) -> Optional[dict]:
    """Curves [3,32] (p10/p50/p90) + cutoff date -> date strings.

    Returns None when the p50 curve never crosses (censored): serve widens to
    at least legacy width or abstains; eval scores it as a miss with max error.
    Ovulation = onset - 14d (confirmation is retrospective; Zhu 2.7+-1.9d).
    """
    try:
        y, m, d = int(str(cutoff)[0:4]), int(str(cutoff)[5:7]), int(str(cutoff)[8:10])
        anchor = date(y, m, d)
    except Exception:
        return None
    if len(curves) != 3 or any(len(c) != HORIZON for c in curves):
        return None
    p10_h = first_crossing(list(curves[0]), threshold)
    p50_h = first_crossing(list(curves[1]), threshold)
    p90_h = first_crossing(list(curves[2]), threshold)
    if p50_h is None:
        return None
    # Interval must bracket p50; fall back to +-3d when a tail never crosses.
    lo = p10_h if p10_h is not None and p10_h <= p50_h else max(1, p50_h - 3)
    hi = p90_h if p90_h is not None and p90_h >= p50_h else p50_h + 3
    nxt = anchor + timedelta(days=p50_h)
    ov = nxt - timedelta(days=14)
    return {
        "days_to_onset_p50": p50_h,
        "days_to_onset_p10": lo,
        "days_to_onset_p90": hi,
        "next_onset": nxt.isoformat(),
        "ovulation_date": ov.isoformat(),
        "fertile_start": (ov - timedelta(days=5)).isoformat(),
        "fertile_end": ov.isoformat(),
        "p10": (anchor + timedelta(days=lo)).isoformat(),
        "p50": nxt.isoformat(),
        "p90": (anchor + timedelta(days=hi)).isoformat(),
    }
