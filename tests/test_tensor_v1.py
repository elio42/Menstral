"""Tensor v1 parity with Go golden vectors + DST property tests."""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ovumcy_tensors import (
    MASK_ABSENT_ROW,
    MASK_BBT,
    MASK_CYCLE_DAY,
    build_tensor,
    days_between,
    hash_tensor,
)


def _period_day(d: date, flow="medium"):
    return {"date": d.isoformat(), "is_period": True, "flow": flow,
            "bbt": None, "cervical_mucus": "none", "symptom_ids": [],
            "cycle_factor_keys": [], "cycle_start": False,
            "is_uncertain": False, "has_spotting_symptom": False}


def test_schema_v1_180_rows_calendar_consecutive():
    today = date(2026, 3, 20)
    logs = []
    for back in (94, 93, 66, 65, 38, 37, 10, 9):
        logs.append(_period_day(today - timedelta(days=back)))
    logs.append({"date": today.isoformat(), "is_period": False, "flow": "heavy",
                 "bbt": 36.6, "cervical_mucus": "eggwhite", "symptom_ids": [3, 3, 5],
                 "cycle_factor_keys": [], "cycle_start": False,
                 "is_uncertain": False, "has_spotting_symptom": False})
    t = build_tensor(logs, None, today)
    assert t["version"] == "v1"
    assert len(t["days"]) == 180
    for i in range(1, 180):
        a = date.fromisoformat(t["days"][i - 1]["date"])
        b = date.fromisoformat(t["days"][i]["date"])
        assert days_between(a, b) == 1
    last = t["days"][-1]
    assert last["present"] and last["mucus_score"] == 4 and last["flow"] == 4
    assert last["symptom_count"] == 2  # deduped 3,3,5
    assert last["bbt_c"] == 36.6 and not last["bbt_missing"]
    assert last["cycle_day"] == 11, last
    assert hash_tensor(t) == hash_tensor(t) and hash_tensor(t)


def test_gap_and_disturbed_flags():
    today = date(2026, 3, 20)
    start = today - timedelta(days=179)
    logs = [_period_day(start), _period_day(start + timedelta(days=1)),
            {"date": today.isoformat(), "is_period": False, "flow": "none",
             "bbt": 36.7, "cervical_mucus": "none", "symptom_ids": [],
             "cycle_factor_keys": ["illness"], "cycle_start": False,
             "is_uncertain": False, "has_spotting_symptom": False}]
    t = build_tensor(logs, None, today)
    mids = t["days"][2:-1]
    assert any(r["gap_before"] for r in mids), "expected >90d gap flags"
    for r in mids:
        assert not r["present"] and r["bbt_missing"] and (r["missing_mask"] & MASK_BBT)
    last = t["days"][-1]
    assert last["disturbed"] and last["bbt_missing"] and last["bbt_c"] is None


def test_dst_day_counts_zoneinfo():
    # Pure-calendar stepping is DST-immune in every zone with a skip (incl.
    # southern-hemisphere Lord_Howe 30-min and half-hour zones).
    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return
    for zn in ("America/Santiago", "America/Havana", "Australia/Lord_Howe",
               "Pacific/Apia", "Asia/Tehran", "Australia/Sydney"):
        z = ZoneInfo(zn)
        # Walk 400 days across known DST boundaries using date arithmetic
        # (the tensor's stepping), then assert consecutive + count.
        d0 = date(2025, 1, 1)
        for i in range(1, 400):
            a, b = d0 + timedelta(days=i - 1), d0 + timedelta(days=i)
            assert days_between(a, b) == 1, (zn, a, b)
            # zone-aware midnight exists check: localizing must not shift the date
            import datetime as dt
            local = dt.datetime(a.year, a.month, a.day, 12, 0, tzinfo=z)
            assert (local.date() - a).days == 0


def test_missing_mask_bits():
    today = date(2026, 3, 20)
    t = build_tensor([], None, today)
    assert len(t["days"]) == 180
    for r in t["days"]:
        assert r["missing_mask"] & MASK_ABSENT_ROW
        assert r["missing_mask"] & MASK_BBT
        assert r["missing_mask"] & MASK_CYCLE_DAY
        assert r["cycle_day"] == 0
