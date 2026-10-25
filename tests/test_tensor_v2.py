"""Tensor v2 parity — first 12 cols byte-identical to v1 + peak cols."""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ovumcy_tensors import build_tensor, to_model_matrix  # noqa: E402
from ovumcy_tensors.tensor_v2 import (  # noqa: E402
    FEATURE_COLUMNS,
    NUM_FEATURES,
    build_tensor_v2,
    to_model_matrix_v2,
)


def _period_day(d: date, flow="medium", peak_day=False, peak_missing=False):
    return {"date": d.isoformat(), "is_period": True, "flow": flow,
            "bbt": None, "cervical_mucus": "none", "symptom_ids": [],
            "cycle_factor_keys": [], "cycle_start": False,
            "is_uncertain": False, "has_spotting_symptom": False,
            "peak_day": peak_day, "peak_missing": peak_missing}


def test_v2_feature_order_and_width():
    assert NUM_FEATURES == 14
    assert FEATURE_COLUMNS[:12] == ["bleed", "flow_norm", "bbt_z", "bbt_missing",
                                    "mucus_norm", "symptom_log", "disturbed",
                                    "cycle_day_norm", "cycle_day_missing",
                                    "gap_before", "present", "absent"]
    assert FEATURE_COLUMNS[12:] == ["peak", "peak_missing"]


def test_v2_first12_identical_to_v1():
    today = date(2026, 3, 20)
    logs = [_period_day(today - timedelta(days=10), peak_day=False),
            _period_day(today - timedelta(days=9), peak_day=True),
            {"date": today.isoformat(), "is_period": False, "flow": "none",
             "bbt": None, "cervical_mucus": "none", "symptom_ids": [],
             "cycle_factor_keys": [], "cycle_start": False,
             "is_uncertain": False, "has_spotting_symptom": False,
             "peak_day": False, "peak_missing": False}]
    v1 = build_tensor(logs, None, today)
    v2 = build_tensor_v2(logs, None, today)
    assert v2["version"] == "v2" and len(v2["days"]) == 180
    for r1, r2 in zip(v1["days"], v2["days"]):
        for k in ("date", "present", "bleed", "flow", "cycle_day", "gap_before"):
            assert r1[k] == r2[k], (k, r1, r2)
    m1 = to_model_matrix(v1)
    m2 = to_model_matrix_v2(v2)
    for a, b in zip(m1, m2):
        assert a == b[:12]
        assert len(b) == 14


def test_v2_peak_values_and_absent_missing():
    today = date(2026, 3, 20)
    peak_day = today - timedelta(days=5)
    logs = [_period_day(today - timedelta(days=40)),
            _period_day(peak_day, peak_day=True),
            {"date": today.isoformat(), "is_period": False, "flow": "none",
             "bbt": None, "cervical_mucus": "none", "symptom_ids": [],
             "cycle_factor_keys": [], "cycle_start": False,
             "is_uncertain": False, "has_spotting_symptom": False,
             "peak_day": False, "peak_missing": False}]
    v2 = build_tensor_v2(logs, None, today)
    m2 = to_model_matrix_v2(v2)
    rows = {r["date"]: (r, m) for r, m in zip(v2["days"], m2)}
    # peak day present with observed position
    assert rows[peak_day.isoformat()][0]["peak"] == 1
    assert rows[peak_day.isoformat()][1][12:] == [1.0, 0.0]
    # observed non-peak day
    assert rows[today.isoformat()][1][12:] == [0.0, 0.0]
    # absent day: unknown
    assert rows[(today - timedelta(days=3)).isoformat()][1][12:] == [0.0, 1.0]
    # estimated position propagates peak_missing=1 on the peak day
    logs2 = [_period_day(peak_day, peak_day=True, peak_missing=True)]
    v2b = build_tensor_v2(logs2, None, peak_day)
    m2b = to_model_matrix_v2(v2b)
    hit = [m for r, m in zip(v2b["days"], m2b) if r["date"] == peak_day.isoformat()][0]
    assert hit[12:] == [1.0, 1.0]
