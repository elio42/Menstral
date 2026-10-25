"""Daily tensor v1 — Python mirror of Go BuildPredictorTensor.

Normative: docs/predictor-contract.md in ovumcy-web +
internal/services/predictor_tensor.go. The two builders must stay
byte-identical in rows: same 180-day window, same gap/mask/bleed/flow/mucus
rules, same CycleBoundaries. Model code reads the float matrix via
to_model_matrix(); the wire JSON uses the same field names as Go.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional

VERSION = "v1"
WINDOW_DAYS = 180
SKIP_GAP_DAYS = 90

MASK_ABSENT_ROW = 1 << 0
MASK_BBT = 1 << 1
MASK_CYCLE_DAY = 1 << 2

MIN_BBT_C = 34.0
MAX_BBT_C = 43.0

# Frozen v1 feature order for the model matrix [180, 12].
FEATURE_COLUMNS = [
    "bleed",            # 0/1 non-spotting period day
    "flow_norm",        # flow 0-4 / 4
    "bbt_z",            # (bbt - mean)/std, 0 when missing
    "bbt_missing",      # 0/1
    "mucus_norm",       # mucus 0-4 / 4
    "symptom_log",      # log1p(symptom_count)/log1p(8)
    "disturbed",        # illness or sleep_disruption 0/1
    "cycle_day_norm",   # cycle_day / 35, 0 when unknown
    "cycle_day_missing",  # 0/1
    "gap_before",       # 0/1 (>90d silence)
    "present",          # 0/1
    "absent",           # 1-present (explicit for attention masks)
]
NUM_FEATURES = len(FEATURE_COLUMNS)

FLOW_VALUES = {"spotting": 1, "light": 2, "medium": 3, "heavy": 4}
MUCUS_VALUES = {"dry": 1, "moist": 2, "creamy": 3, "eggwhite": 4}


def normalize_flow(flow: Any) -> str:
    s = str(flow or "").strip().lower()
    if s in ("spotting", "light", "medium", "heavy"):
        return s
    return "none"


def normalize_mucus(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s in ("dry", "moist", "creamy", "eggwhite"):
        return s
    return "none"


def flow_score(flow: Any) -> int:
    return FLOW_VALUES.get(normalize_flow(flow), 0)


def mucus_score(value: Any) -> int:
    return MUCUS_VALUES.get(normalize_mucus(value), 0)


def is_valid_bbt(v: Any) -> bool:
    if v is None:
        return True
    try:
        f = float(v)
    except (TypeError, ValueError):
        return False
    return MIN_BBT_C <= f <= MAX_BBT_C


def is_disturbed(log: dict) -> bool:
    keys = log.get("cycle_factor_keys") or log.get("cycleFactorKeys") or []
    for k in keys:
        ks = str(k).strip().lower()
        if ks in ("illness", "sleep_disruption"):
            return True
    return False


def is_spotting_day(log: dict) -> bool:
    flow = normalize_flow(log.get("flow"))
    if flow == "spotting":
        return True
    # Owner's Spotting symptom is the only bleeding signal (no flow chosen).
    if log.get("has_spotting_symptom") and flow in ("none", ""):
        return True
    return False


def unique_symptom_ids(ids: Any) -> list:
    if not ids:
        return []
    seen: set = set()
    out: list = []
    for i in ids:
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


def parse_day(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not hasattr(value, "hour"):
        return value
    if hasattr(value, "date"):
        try:
            return value.date()  # datetime -> date (calendar components as stored)
        except Exception:
            pass
    s = str(value).strip()[:10]
    try:
        y, m, d = int(s[0:4]), int(s[5:7]), int(s[8:10])
        return date(y, m, d)
    except Exception:
        return None


def days_between(a: date, b: date) -> int:
    """Signed calendar days a -> b. Pure date arithmetic: DST-immune by construction."""
    return (b - a).days


# --- Cycle boundaries (mirror of Go cycle_boundaries.go) ---

def _period_days_of(logs: list[dict]) -> list[dict]:
    days: list[dict] = []
    for log in sorted(logs, key=lambda l: parse_day(l.get("date")) or date.min):
        if not log.get("is_period"):
            continue
        d = parse_day(log.get("date"))
        if d is None:
            continue
        entry = {
            "day": d,
            "spotting": is_spotting_day(log),
            "explicit": bool(log.get("cycle_start")) and not bool(log.get("is_uncertain")),
            "uncertain": bool(log.get("cycle_start")) and bool(log.get("is_uncertain")),
        }
        if days and days[-1]["day"] == d:
            days[-1]["spotting"] = days[-1]["spotting"] and entry["spotting"]
            days[-1]["explicit"] = days[-1]["explicit"] or entry["explicit"]
            days[-1]["uncertain"] = days[-1]["uncertain"] or entry["uncertain"]
            continue
        days.append(entry)
    return days


def _cluster_days(days: list[dict]) -> list[dict]:
    clusters: list[dict] = []
    for e in days:
        if clusters:
            last = clusters[-1]
            if days_between(last["end"], e["day"]) - 1 < 5:
                last["days"].append(e)
                if e["day"] > last["end"]:
                    last["end"] = e["day"]
                continue
        clusters.append({"start": e["day"], "end": e["day"], "days": [e]})
    return clusters


def _cluster_boundary(cluster: dict, today: Optional[date]):
    saw_uncertain = False
    for e in cluster["days"]:
        if e["spotting"]:
            continue
        if e["explicit"]:
            return e["day"], True
        if e["uncertain"]:
            saw_uncertain = True
    if saw_uncertain:
        return None, False
    for i, e in enumerate(cluster["days"]):
        if e["spotting"]:
            continue
        if i + 1 < len(cluster["days"]):
            nxt = cluster["days"][i + 1]
            if not nxt["spotting"] and days_between(e["day"], nxt["day"]) == 1:
                return e["day"], True
        if today is not None:
            since = days_between(e["day"], today)
            if since == 0 or since == 1:
                return e["day"], True
    return None, False


def cycle_boundaries(logs: list[dict], last_period_start: Any, today: date) -> list[date]:
    """All cycle starts ascending. Mirrors Go CycleBoundaries."""
    days = _period_days_of(logs)
    onboarding = parse_day(last_period_start)
    if onboarding is not None and days_between(today, onboarding) <= 0:
        # Join as explicit non-spotting day.
        found = False
        for e in days:
            if e["day"] == onboarding:
                e["spotting"] = False
                e["explicit"] = True
                found = True
        if not found:
            pos = len(days)
            for i, e in enumerate(days):
                if e["day"] > onboarding:
                    pos = i
                    break
            days.insert(pos, {"day": onboarding, "spotting": False, "explicit": True, "uncertain": False})
    clusters = _cluster_days(days)
    starts: list[date] = []
    for c in clusters:
        s, opens = _cluster_boundary(c, today)
        if opens and s is not None:
            starts.append(s)
    return sorted(starts)


def latest_boundary_on_or_before(starts: list[date], day: date) -> Optional[date]:
    for s in reversed(starts):
        if days_between(day, s) <= 0:
            return s
    return None


# --- Tensor builder ---

@dataclass
class DayRow:
    date: str
    present: bool
    bleed: int = 0
    flow: int = 0
    bbt_c: Optional[float] = None
    bbt_missing: bool = True
    mucus_score: int = 0
    symptom_count: int = 0
    disturbed: bool = False
    cycle_day: int = 0
    gap_before: bool = False
    missing_mask: int = 0

    def to_wire(self) -> dict:
        return {
            "date": self.date,
            "present": self.present,
            "bleed": self.bleed,
            "flow": self.flow,
            "bbt_c": self.bbt_c,
            "bbt_missing": self.bbt_missing,
            "mucus_score": self.mucus_score,
            "symptom_count": self.symptom_count,
            "disturbed": self.disturbed,
            "cycle_day": self.cycle_day,
            "gap_before": self.gap_before,
            "missing_mask": self.missing_mask,
        }


def build_tensor(
    logs: list[dict],
    last_period_start: Any,
    today: Any,
) -> dict:
    """Build v1 tensor over [today-179, today]. Mirrors Go BuildPredictorTensor.

    Latest same-day write wins. Days after today are dropped (calendar-day
    comparison, DST-safe). Absent rows carry only flags + cycle_day.
    """
    today_d = parse_day(today)
    assert today_d is not None, "today must be a valid date"
    # Filter logs not after today (calendar-day comparison).
    filt: list[dict] = []
    for log in logs or []:
        d = parse_day(log.get("date"))
        if d is None:
            continue
        if days_between(today_d, d) > 0:
            continue
        filt.append(log)
    boundaries = cycle_boundaries(filt, last_period_start, today_d)

    by_day: dict[str, dict] = {}
    for log in sorted(filt, key=lambda l: parse_day(l.get("date")) or date.min):
        d = parse_day(log.get("date"))
        assert d is not None
        by_day[d.isoformat()] = log  # latest wins

    start = today_d - timedelta(days=WINDOW_DAYS - 1)
    rows: list[DayRow] = []
    days_since_present = -1
    day = start
    while days_between(day, today_d) >= 0:
        key = day.isoformat()
        entry = by_day.get(key)
        if entry is None:
            anchor = latest_boundary_on_or_before(boundaries, day)
            cd = days_between(anchor, day) + 1 if anchor else 0
            mask = MASK_ABSENT_ROW | MASK_BBT
            if cd <= 0:
                mask |= MASK_CYCLE_DAY
            rows.append(DayRow(
                date=key, present=False, bbt_missing=True,
                cycle_day=cd,
                gap_before=(days_since_present > SKIP_GAP_DAYS),
                missing_mask=mask,
            ))
            if days_since_present >= 0:
                days_since_present += 1
        else:
            days_since_present = 0
            mask = 0
            bleed = 1 if (entry.get("is_period") and not is_spotting_day(entry)) else 0
            fl = flow_score(entry.get("flow"))
            bbt_raw = entry.get("bbt")
            bbt_val: Optional[float] = None
            try:
                bbt_f = float(bbt_raw) if bbt_raw is not None else None
            except (TypeError, ValueError):
                bbt_f = None
            if bbt_f is not None and is_valid_bbt(bbt_f) and not is_disturbed(entry):
                bbt_val = bbt_f
            else:
                mask |= MASK_BBT
            muc = mucus_score(entry.get("cervical_mucus"))
            sym = len(unique_symptom_ids(entry.get("symptom_ids") or entry.get("symptomIds") or []))
            dist = is_disturbed(entry)
            anchor = latest_boundary_on_or_before(boundaries, day)
            cd = days_between(anchor, day) + 1 if anchor else 0
            if cd <= 0:
                mask |= MASK_CYCLE_DAY
            rows.append(DayRow(
                date=key, present=True, bleed=bleed, flow=fl,
                bbt_c=bbt_val, bbt_missing=(bbt_val is None),
                mucus_score=muc, symptom_count=sym, disturbed=dist,
                cycle_day=cd, gap_before=False, missing_mask=mask,
            ))
        day += timedelta(days=1)
    return {"version": VERSION, "days": [r.to_wire() for r in rows]}


def hash_tensor(tensor: dict) -> str:
    raw = json.dumps(tensor, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def to_model_matrix(
    tensor: dict,
    bbt_mean: float = 36.80,
    bbt_std: float = 0.35,
    cycle_day_scale: float = 35.0,
) -> "list[list[float]]":
    """Wire rows -> z-normalized float matrix [180, 12]. No torch needed."""
    import math

    mat: list[list[float]] = []
    for r in tensor["days"]:
        bleed = float(r.get("bleed", 0))
        flow_n = float(r.get("flow", 0)) / 4.0
        bbt = r.get("bbt_c")
        if bbt is None:
            bbt_z, bbt_m = 0.0, 1.0
        else:
            bbt_z, bbt_m = (float(bbt) - bbt_mean) / bbt_std, 0.0
        muc_n = float(r.get("mucus_score", 0)) / 4.0
        sym = math.log1p(float(r.get("symptom_count", 0))) / math.log1p(8.0)
        dist = 1.0 if r.get("disturbed") else 0.0
        cd = int(r.get("cycle_day", 0) or 0)
        cd_n = (cd / cycle_day_scale) if cd > 0 else 0.0
        cd_m = 0.0 if cd > 0 else 1.0
        gap = 1.0 if r.get("gap_before") else 0.0
        pres = 1.0 if r.get("present") else 0.0
        mat.append([bleed, flow_n, bbt_z, bbt_m, muc_n, sym, dist,
                    cd_n, cd_m, gap, pres, 1.0 - pres])
    return mat
