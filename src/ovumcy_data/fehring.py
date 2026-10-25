"""Fehring loader — licensed Marquette CSV -> daily timelines (local-only).

Chains each client's cycles in CycleNumber order into one daily timeline.
Deterministic under seed; sorted output. Torch-free.

Rules (decided in artifacts/eda/fehring_eda.md §3–§4, documented here):
  FLOW_MAP: blank -> none (no bleed), 1 -> light, 2 -> medium, 3 -> heavy.
    Score 1 counts as a bleed day (it counts toward LengthofMenses), so it is
    NEVER mapped to spotting. No spotting rows are ever emitted.
  PEAK_RULE:
    CycleWithPeakorNot == 0 -> no peak day (peak_day=0, peak_missing=0).
    == 1 and 1 <= EDO <= LengthofCycle -> peak_day=1 on cycle day EDO.
    == 1 and EDO blank -> peak_day=1 on FirstDayofHigh + 4 (median offset),
      clamped to [1, LengthofCycle]; that cycle's peak_missing=1.
    == 1 and both blank -> peak_day=0 everywhere, peak_missing=1.
  IRREGULAR_RULE: client irregular iff cycle-length sd > 3.0 (>=3 cycles)
    else range > 5. Otherwise regular.
  FLAGS: unusual_flag=1 when UnusualBleeding==1 or PhasesBleeding in {1,2,3};
    postpartum_flag=1 when ReproductiveCategory not in ("", "0") or
    Breastfeeding not in ("", "0"). Both carried per day for stratification;
    no rows are excluded here (Phase B training excludes postpartum cycles).
  DEDUP: group by (ClientID, CycleNumber), keep the first row.
  DATES: synthetic calendar anchor only (the CSV carries no dates).
    base 2020-01-01 + per-client stagger in [0,30]d from Random(seed).
    Same seed + same file = same timelines. Clients are independent users;
    absolute dates carry no meaning and never leave this machine.
  PRESENT: every reconstructed day is marked present=True with is_period /
    flow from the score. They are observed coverage, not owner-logged days —
    marking them absent trips the >90d gap flags and destroys timelines.
  NEVER IN LOGS: BBT (always None/masked), intercourse columns (analysis
    only), age/BMI/demographics (manifest only).

Each daily log is a tensor-builder-compatible dict plus Fehring extras
(peak_day, peak_missing, unusual_flag, postpartum_flag, fehring_client,
fehring_cycle, cycle_day_true) that v1 ignores and v2 consumes.
"""
from __future__ import annotations

import csv
import random
from datetime import date, timedelta

FLOW_MAP = {"": "none", "1": "light", "2": "medium", "3": "heavy"}
PEAK_FALLBACK_OFFSET = 4
BASE_ANCHOR = date(2020, 1, 1)

_MENSES_BY_DAY = {
    1: "MensesScoreDayOne", 2: "MensesScoreDayTwo", 3: "MensesScoreDayThree",
    4: "MensesScoreDayFour", 5: "MensesScoreDayFive", 6: "MensesScoreDaySix",
    7: "MensesScoreDaySeven", 8: "MensesScoreDayEight", 9: "MensesScoreDayNine",
    10: "MensesScoreDayTen", 11: "MensesScoreDay11", 12: "MensesScoreDay12",
    13: "MensesScoreDay13", 14: "MensesScoreDay14", 15: "MensesScoreDay15",
}


def _blank(v) -> bool:
    return v is None or str(v).strip() == ""


def _num(v):
    try:
        return float(str(v).strip())
    except Exception:
        return None


def _int(v, default=None):
    n = _num(v)
    if n is None:
        return default
    try:
        return int(n)
    except Exception:
        return default


def _dedup_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Group by (ClientID, CycleNumber), keep first. Returns (kept, n_dropped)."""
    seen: dict[tuple[str, str], dict] = {}
    dropped = 0
    for r in rows:
        key = ((r.get("ClientID") or "").strip(), (r.get("CycleNumber") or "").strip())
        if key in seen:
            dropped += 1
            continue
        seen[key] = r
    return list(seen.values()), dropped


def _peak_day_for_cycle(row: dict, length: int) -> tuple[int | None, int]:
    """Returns (peak_cycle_day or None, peak_missing 0/1)."""
    flag = (row.get("CycleWithPeakorNot") or "").strip()
    edo = _int(row.get("EstimatedDayofOvulation"))
    fh = _int(row.get("FirstDayofHigh"))
    if flag == "0":
        return None, 0
    if flag != "1":
        return None, 1  # unknown flag: treat as missing
    if edo is not None and length and 1 <= edo <= length:
        return edo, 0
    if fh is not None and length:
        est = fh + PEAK_FALLBACK_OFFSET
        est = max(1, min(int(length), est))
        return est, 1
    return None, 1


def _client_irregular(lengths: list[int]) -> bool:
    if len(lengths) >= 3:
        import statistics

        return statistics.pstdev(lengths) > 3.0
    if len(lengths) >= 2:
        return max(lengths) - min(lengths) > 5
    return False


def load_fehring_users(csv_path: str, seed: int = 42,
                       anchor: date = BASE_ANCHOR) -> dict:
    """CSV -> {users, stats}. Deterministic under seed; sorted by client id."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    kept, n_dropped = _dedup_rows(rows)

    by_client: dict[str, list[dict]] = {}
    for r in kept:
        cid = (r.get("ClientID") or "").strip()
        by_client.setdefault(cid, []).append(r)
    for cid in by_client:
        by_client[cid].sort(key=lambda r: _int(r.get("CycleNumber"), 0))

    rng = random.Random(seed)
    # Per-client stagger drawn in sorted-client order: deterministic per seed.
    staggers = {cid: rng.randint(0, 30) for cid in sorted(by_client)}

    users: list[dict] = []
    n_days = 0
    n_peak_cycles = 0
    n_unusual = 0
    n_postpartum = 0
    for cid in sorted(by_client):
        cycles = by_client[cid]
        lengths: list[int] = []
        for r in cycles:
            L = _int(r.get("LengthofCycle"))
            if L is not None and L > 0:
                lengths.append(L)
        irregular = _client_irregular(lengths)
        group = (cycles[0].get("Group") or "").strip() if cycles else ""
        start = anchor + timedelta(days=staggers[cid])
        logs: list[dict] = []
        cursor = start
        for r in cycles:
            L = _int(r.get("LengthofCycle"))
            if L is None or L <= 0:
                continue
            peak_cd, peak_missing_cycle = _peak_day_for_cycle(r, L)
            if (r.get("CycleWithPeakorNot") or "").strip() == "1":
                n_peak_cycles += 1
            unusual = (
                (r.get("UnusualBleeding") or "").strip() == "1"
                or (r.get("PhasesBleeding") or "").strip() in ("1", "2", "3")
            )
            postpartum = (
                (r.get("ReproductiveCategory") or "").strip() not in ("", "0")
                or (r.get("Breastfeeding") or "").strip() not in ("", "0")
            )
            if unusual:
                n_unusual += 1
            if postpartum:
                n_postpartum += 1
            cyc_no = _int(r.get("CycleNumber"), 0)
            for cd in range(1, L + 1):
                col = _MENSES_BY_DAY.get(cd)
                score = (r.get(col) or "").strip() if col else ""
                flow = FLOW_MAP.get(score, "none")
                is_period = flow != "none"
                d = cursor + timedelta(days=cd - 1)
                logs.append({
                    "date": d.isoformat(),
                    "is_period": is_period,
                    "flow": flow,
                    "bbt": None,
                    "cervical_mucus": "none",
                    "symptom_ids": [],
                    "cycle_factor_keys": [],
                    "cycle_start": bool(cd == 1 and is_period),
                    "is_uncertain": False,
                    "has_spotting_symptom": False,
                    # Fehring extras (v1 ignores, v2 consumes peak_*):
                    "peak_day": bool(peak_cd is not None and cd == peak_cd),
                    "peak_missing": bool(peak_missing_cycle),
                    "unusual_flag": bool(unusual),
                    "postpartum_flag": bool(postpartum),
                    "fehring_client": cid,
                    "fehring_cycle": cyc_no,
                    "cycle_day_true": cd,
                })
            cursor += timedelta(days=L)
        n_days += len(logs)
        users.append({
            "user_id": f"fehring-{cid}",
            "client_id": cid,
            "irregular": bool(irregular),
            "group": group,
            "logs": logs,
            "lengths": lengths,
        })

    users.sort(key=lambda u: u["user_id"])
    stats = {
        "seed": seed,
        "csv_rows": len(rows),
        "rows_kept": len(kept),
        "rows_dropped_dup": n_dropped,
        "n_clients": len(users),
        "n_days": n_days,
        "n_peak_cycles": n_peak_cycles,
        "n_unusual_cycles": n_unusual,
        "n_postpartum_cycles": n_postpartum,
    }
    return {"users": users, "stats": stats}
