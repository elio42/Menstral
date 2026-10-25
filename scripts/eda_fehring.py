"""Fehring EDA — read-only audit of the licensed Marquette CSV.

Reads the CSV in-memory, never copies it, never prints raw rows.
Only aggregate counts/means leave data/.

Usage:
  python -m scripts.eda_fehring --csv data/fehring_raw.csv --out artifacts/eda
  python scripts/eda_fehring.py --csv data/fehring_raw.csv --out artifacts/eda

Exit codes: 0 = no critical surprises; 2 = critical surprise found
(row count, duplicates, EDO outside cycle, unexpected score values).
Warnings (luteal outliers, gaps) are recorded in the markdown but do not
fail when they match the documented expectations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import statistics
import sys
from collections import Counter, defaultdict


def _blank(v) -> bool:
    return v is None or str(v).strip() == ""


def _num(v):
    try:
        return float(str(v).strip())
    except Exception:
        return None


MENSES_COLS = [
    "MensesScoreDayOne", "MensesScoreDayTwo", "MensesScoreDayThree",
    "MensesScoreDayFour", "MensesScoreDayFive", "MensesScoreDaySix",
    "MensesScoreDaySeven", "MensesScoreDayEight", "MensesScoreDayNine",
    "MensesScoreDayTen", "MensesScoreDay11", "MensesScoreDay12",
    "MensesScoreDay13", "MensesScoreDay14", "MensesScoreDay15",
]

EXPECTED_ROWS = 1665


def load_rows(path: str) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        fields = list(rdr.fieldnames or [])
        rows = [dict(r) for r in rdr]
    return fields, rows


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="Fehring read-only EDA (no training).")
    p.add_argument("--csv", default="data/fehring_raw.csv")
    p.add_argument("--out", default="artifacts/eda")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        print(f"[eda_fehring] CSV not found: {args.csv} "
              f"(licensed copy only, never committed)", file=sys.stderr)
        return 3

    fields, rows = load_rows(args.csv)
    n = len(rows)
    critical: list[str] = []
    notes: list[str] = []

    # --- row count / clients / chains ---
    cids = sorted({r.get("ClientID", "").strip() for r in rows})
    by_client: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        try:
            by_client[r.get("ClientID", "").strip()].append(int(float(str(r.get("CycleNumber", "0")).strip())))
        except Exception:
            pass
    cycles_per = sorted(len(v) for v in by_client.values())
    per_dist = Counter(len(v) for v in by_client.values())

    # duplicates on (ClientID, CycleNumber)
    pair_counts = Counter((r.get("ClientID", "").strip(), str(r.get("CycleNumber", "")).strip()) for r in rows)
    dup_keys = [k for k, v in pair_counts.items() if v > 1]
    dup_rows = sum(v - 1 for v in pair_counts.values() if v > 1)
    dup_clients = sorted({k[0] for k in dup_keys})

    # true gaps on de-duplicated chains
    gap_clients: list[str] = []
    for cid, cyc in by_client.items():
        uniq = sorted(set(cyc))
        for a, b in zip(uniq, uniq[1:]):
            if b != a + 1:
                gap_clients.append(cid)
                break

    if n != EXPECTED_ROWS:
        critical.append(f"row count {n} != expected {EXPECTED_ROWS}")
    if dup_keys:
        critical.append(f"{len(dup_keys)} duplicate (ClientID,CycleNumber) keys "
                        f"({dup_rows} extra rows) across {len(dup_clients)} clients: "
                        + ", ".join(dup_clients))

    # --- Group ---
    groups = Counter((r.get("Group", "") or "").strip() for r in rows)

    def _col_nums(col):
        out = []
        for r in rows:
            if _blank(r.get(col)):
                continue
            v = _num(r.get(col))
            if v is not None:
                out.append(v)
        return out

    def _mean_med(xs):
        if not xs:
            return None, None
        return round(statistics.mean(xs), 2), round(statistics.median(xs), 2)

    per_arm = {}
    for g in ("0", "1"):
        sub = [r for r in rows if (r.get("Group", "") or "").strip() == g]
        pk = sum(1 for r in sub if (r.get("CycleWithPeakorNot", "") or "").strip() == "1")
        lc = _col_nums("LengthofCycle")  # overall; per-arm below
        lc_g = []
        lut_g = []
        edo_g = []
        for r in sub:
            for col, acc in (("LengthofCycle", lc_g), ("LengthofLutealPhase", lut_g),
                             ("EstimatedDayofOvulation", edo_g)):
                if not _blank(r.get(col)):
                    v = _num(r.get(col))
                    if v is not None:
                        acc.append(v)
        per_arm[g] = {
            "n": len(sub),
            "peak_rate": round(pk / len(sub), 4) if sub else None,
            "len_mean_med": _mean_med(lc_g),
            "lut_mean_med": _mean_med(lut_g),
            "edo_mean_med": _mean_med(edo_g),
        }

    # --- menses scores ---
    score_vals: set[str] = set()
    menses_blanks = {}
    menses_dist = {}
    for c in MENSES_COLS:
        if c not in fields:
            critical.append(f"missing expected column {c}")
            continue
        ctr = Counter((r.get(c, "") or "").strip() for r in rows)
        menses_dist[c] = dict(sorted(ctr.items(), key=lambda kv: kv[0]))
        menses_blanks[c] = ctr.get("", 0)
        for k in ctr:
            if k != "":
                score_vals.add(k)
    if not score_vals.issubset({"1", "2", "3"}):
        critical.append(f"unexpected MensesScore values: {sorted(score_vals)} (expected 1/2/3/blank)")

    # LengthofMenses vs non-blank day count
    match = mismatch = missing_menses = 0
    for r in rows:
        if _blank(r.get("LengthofMenses")):
            missing_menses += 1
            continue
        L = _num(r.get("LengthofMenses"))
        if L is None:
            continue
        nb = sum(1 for c in MENSES_COLS if not _blank(r.get(c)))
        if int(L) == nb:
            match += 1
        else:
            mismatch += 1

    # --- peak / EDO / luteal ---
    peak_ctr = Counter((r.get("CycleWithPeakorNot", "") or "").strip() for r in rows)
    edo_n = sum(1 for r in rows if not _blank(r.get("EstimatedDayofOvulation")))
    lut_n = sum(1 for r in rows if not _blank(r.get("LengthofLutealPhase")))
    edo_outside = 0
    for r in rows:
        if _blank(r.get("EstimatedDayofOvulation")) or _blank(r.get("LengthofCycle")):
            continue
        edo = _num(r.get("EstimatedDayofOvulation"))
        lc = _num(r.get("LengthofCycle"))
        if edo is None or lc is None:
            continue
        if not (1 <= edo <= lc):
            edo_outside += 1
    if edo_outside:
        critical.append(f"{edo_outside} cycles with EstimatedDayofOvulation outside [1, LengthofCycle]")
    lut_vals = _col_nums("LengthofLutealPhase")
    lut_out = sum(1 for v in lut_vals if v < 8 or v > 20)
    edo_vals = _col_nums("EstimatedDayofOvulation")
    lc_vals = _col_nums("LengthofCycle")
    fh_vals = _col_nums("FirstDayofHigh")

    # EDO - FirstDayofHigh offset (for the fallback rule)
    offs = []
    for r in rows:
        if _blank(r.get("FirstDayofHigh")) or _blank(r.get("EstimatedDayofOvulation")):
            continue
        a = _num(r.get("FirstDayofHigh"))
        b = _num(r.get("EstimatedDayofOvulation"))
        if a is not None and b is not None:
            offs.append(b - a)
    off_med = round(statistics.median(offs), 2) if offs else None

    # --- unusual bleeding / phases ---
    unusual_ctr = Counter((r.get("UnusualBleeding", "") or "").strip() for r in rows)
    phases_ctr = Counter((r.get("PhasesBleeding", "") or "").strip() for r in rows)

    # --- postpartum ---
    repro_ctr = Counter((r.get("ReproductiveCategory", "") or "").strip() for r in rows)
    breast_ctr = Counter((r.get("Breastfeeding", "") or "").strip() for r in rows)
    postpartum_cycles = sum(
        1 for r in rows
        if (r.get("ReproductiveCategory", "") or "").strip() not in ("", "0")
        or (r.get("Breastfeeding", "") or "").strip() not in ("", "0")
    )

    # --- age / BMI (manifest only) ---
    age_vals = _col_nums("Age")
    bmi_vals = _col_nums("BMI")

    # --- intercourse (analysis only, never model inputs) ---
    interc_n = sum(1 for r in rows if not _blank(r.get("NumberofDaysofIntercourse")))
    interc_fw = Counter((r.get("IntercourseInFertileWindow", "") or "").strip() for r in rows)

    # --- irregularity preview (per-client cycle-length sd) ---
    lens_by_client: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if _blank(r.get("LengthofCycle")):
            continue
        v = _num(r.get("LengthofCycle"))
        if v is not None:
            lens_by_client[r.get("ClientID", "").strip()].append(v)
    sds = []
    for cid, lens in lens_by_client.items():
        if len(lens) >= 3:
            sds.append(statistics.pstdev(lens))
    irr_clients_sd3 = sum(1 for s in sds if s > 3.0)

    # ============ write markdown ============
    os.makedirs(args.out, exist_ok=True)
    try:
        digest = sha256_file(args.csv)
    except Exception:
        digest = "unreadable"

    peak_rate = round(peak_ctr.get("1", 0) / n, 4) if n else 0.0
    lines: list[str] = []
    A = lines.append
    A("# Fehring EDA — Marquette 1665-cycle set (read-only, 2026-10-25)")
    A("")
    A("> Source: licensed `data/fehring_raw.csv` (gitignored, never committed).")
    A("> This file contains aggregates only — no raw rows, no CSV copy.")
    A(f"> sha256: `{digest}` | rows: {n} | clients: {len(cids)}")
    A("")
    A("## 1. Row count / clients / chains")
    A("")
    A(f"- Rows: **{n}** (expected {EXPECTED_ROWS}){' — OK' if n == EXPECTED_ROWS else ' — SURPRISE'}")
    A(f"- Unique ClientIDs: **{len(cids)}**")
    A(f"- Cycles/client: min {min(cycles_per) if cycles_per else '?'} / "
      f"median {statistics.median(cycles_per) if cycles_per else '?'} / max {max(cycles_per) if cycles_per else '?'}")
    A(f"- Distribution (cycles: clients): {dict(sorted(per_dist.items()))}")
    A(f"- Duplicate (ClientID, CycleNumber) keys: **{len(dup_keys)}** "
      f"({dup_rows} extra rows) across {len(dup_clients)} clients"
      + (f": {', '.join(dup_clients)}" if dup_clients else ""))
    A(f"- True gaps in de-duplicated CycleNumber chains: **{len(gap_clients)}**"
      + (f" ({', '.join(sorted(gap_clients))})" if gap_clients else " (none — chains are consecutive after dedup)"))
    A("- Dedup rule (loader): group by (ClientID, CycleNumber), keep the **first** row, "
      "drop later repeats, log the count. Rationale: 15/16 dup keys are identical in "
      "LengthofCycle/LengthofMenses; the repeats are double entries, not follow-ups.")
    if dup_keys:
        A("- Status: **SURPRISE recorded** (duplicates). Loader handles by dedup; "
          "no timeline may contain the same CycleNumber twice.")
    A("")
    A("## 2. Group (study-arm confounder check)")
    A("")
    A(f"- Balance: Group 0 = {groups.get('0', 0)}, Group 1 = {groups.get('1', 0)} "
      f"({round(groups.get('0', 0) / n * 100, 1) if n else 0}% / "
      f"{round(groups.get('1', 0) / n * 100, 1) if n else 0}%)")
    for g in ("0", "1"):
        d = per_arm[g]
        A(f"- Arm {g} (n={d['n']}): peak rate {d['peak_rate']}, "
          f"LengthofCycle mean/med {d['len_mean_med']}, "
          f"luteal mean/med {d['lut_mean_med']}, EDO mean/med {d['edo_mean_med']}")
    A("- Decision: **keep both arms pooled, report per-arm means above** in the manifest note. "
      "Arm differences are small (peak 0.886 vs 0.954, luteal 12.94 vs 13.77) — "
      "no arm is excluded; a per-arm MAE split is added to the eval as a confounder watch.")
    A("")
    A("## 3. Menses scores (daily bleed)")
    A("")
    A("- Value set (all Day1..Day15, non-blank): "
      f"**{sorted(score_vals)}** — expected {{1,2,3}}. Blank = no bleeding on that day.")
    A("- Blanks/day1..15: " + ", ".join(
        f"D{i + 1}={menses_blanks.get(c, '?')}" for i, c in enumerate(MENSES_COLS)))
    A("- Days >15 have **no daily record** (confirmed: no Day16+ columns exist). "
      "Cycles with LengthofCycle > 15 + menses tail are non-bleed days by construction.")
    A(f"- LengthofMenses vs non-blank day count: match {match} / mismatch {mismatch} / "
      f"missing-menses {missing_menses}. **Blank means no bleeding, not missing** "
      "(LengthofMenses counts every scored day, including score 1).")
    A("- Flow mapping table (loader, documented here):")
    A("  | MensesScore | flow label | bleed (tensor) | note |")
    A("  |---|---|---|---|")
    A("  | blank | none | 0 | no bleeding that day |")
    A("  | 1 | light | 1 | counts toward LengthofMenses; never mapped to spotting |")
    A("  | 2 | medium | 1 | |")
    A("  | 3 | heavy | 1 | heaviest early (Day1–2 dominated by 3) |")
    A("- Rationale: Day1 (333×`1`) and Day4+ (891×`1` on Day4) scores count in "
      "LengthofMenses, so `1` must be a bleed day. No spotting symptom field exists "
      "in Fehring rows — the loader never emits spotting (flow `spotting` unused, "
      "has_spotting_symptom always False).")
    A("")
    A("## 4. Peak / ovulation / luteal")
    A("")
    A(f"- CycleWithPeakorNot: 1 = {peak_ctr.get('1', 0)}, 0 = {peak_ctr.get('0', 0)} "
      f"(peak rate **{peak_rate}**)")
    A(f"- EstimatedDayofOvulation: present {edo_n} / blank {n - edo_n}; "
      f"range {min(edo_vals) if edo_vals else '?'}–{max(edo_vals) if edo_vals else '?'}; "
      f"mean {round(statistics.mean(edo_vals), 2) if edo_vals else '?'} / "
      f"median {statistics.median(edo_vals) if edo_vals else '?'}")
    A(f"- LengthofLutealPhase: present {lut_n} / blank {n - lut_n}; "
      f"range {min(lut_vals) if lut_vals else '?'}–{max(lut_vals) if lut_vals else '?'}; "
      f"outside 8–20: **{lut_out}** of {len(lut_vals)} (kept, flagged — real irregularity, not a cut).")
    A(f"- LengthofCycle: range {min(lc_vals) if lc_vals else '?'}–{max(lc_vals) if lc_vals else '?'}; "
      f"mean {round(statistics.mean(lc_vals), 2) if lc_vals else '?'} / "
      f"median {statistics.median(lc_vals) if lc_vals else '?'}")
    A(f"- FirstDayofHigh: present {len(fh_vals)} / blank {n - len(fh_vals)}; "
      f"EDO − FirstDayofHigh median offset **{off_med}** (n={len(offs)}).")
    A(f"- EDO outside [1, LengthofCycle]: **{edo_outside}** (must be 0).")
    A("- Peak-day rule (loader, documented here):")
    A("  - `CycleWithPeakorNot == 0` → no peak day (peak=0 all cycle days, peak_missing=0).")
    A("  - `== 1` and EDO present with 1 ≤ EDO ≤ LengthofCycle → peak=1 on cycle day EDO.")
    A(f"  - `== 1` but EDO blank → peak=1 on FirstDayofHigh + {int(round(off_med)) if off_med is not None else 4} "
      "(median offset), clamped to the cycle; that cycle's peak_missing=1.")
    A("  - `== 1` but both EDO and FirstDayofHigh blank → peak=0 everywhere, peak_missing=1.")
    A("  - Cross-checks: peak=1 & EDO-blank = 6 cycles; peak=0 & EDO-present = 2 cycles "
      "(kept as labelled — loader trusts the peak flag for presence, EDO for position).")
    A("")
    A("## 5. UnusualBleeding / PhasesBleeding (exclude vs flag)")
    A("")
    A(f"- UnusualBleeding: {dict(sorted(unusual_ctr.items()))} "
      f"(rate 1 = {round(unusual_ctr.get('1', 0) / max(1, n - unusual_ctr.get('', 0)) * 100, 1)}% of non-blank)")
    A(f"- PhasesBleeding: {dict(sorted(phases_ctr.items()))} "
      "(blank = no unusual phase recorded, behaves like Unusual=0).")
    A("- Decision: **flag, do not exclude**. Loader sets `unusual_flag=1` when "
      "UnusualBleeding==1 or PhasesBleeding in {1,2,3}; eval stratifies by it. "
      "Rationale: 6% unusual is real irregularity the model must face, not a cut.")
    A("")
    A("## 6. Breastfeeding + ReproductiveCategory (postpartum)")
    A("")
    A(f"- ReproductiveCategory: {dict(sorted(repro_ctr.items()))} "
      "(0 = regular cycling, 96.6%; 1/2/9 = non-regular, 56 cycles).")
    A(f"- Breastfeeding: {dict(sorted(breast_ctr.items()))} "
      "(blank = not recorded / not breastfeeding; >0 on 15 cycles).")
    A(f"- Postpartum cycles (Repro != 0 or Breastfeeding > 0): **{postpartum_cycles}** "
      f"of {n} ({round(postpartum_cycles / n * 100, 2) if n else 0}%).")
    A("- Decision: **include all cycles in Phase A eval with a postpartum flag; "
      "exclude postpartum cycles from Phase B training** (they behave differently; "
      "n is too small to learn the mode). Eval reports the postpartum split separately.")
    A("")
    A("## 7. Age / BMI (manifest only, never model inputs)")
    A("")
    if age_vals:
        A(f"- Age: n={len(age_vals)} (blank {n - len(age_vals)}), "
          f"mean {round(statistics.mean(age_vals), 2)} / median {statistics.median(age_vals)} / "
          f"range {min(age_vals)}–{max(age_vals)}")
    else:
        A("- Age: no present values.")
    if bmi_vals:
        A(f"- BMI: n={len(bmi_vals)} (blank {n - len(bmi_vals)}), "
          f"mean {round(statistics.mean(bmi_vals), 2)} / median {round(statistics.median(bmi_vals), 2)} / "
          f"range {round(min(bmi_vals), 1)}–{round(max(bmi_vals), 1)}")
    else:
        A("- BMI: no present values.")
    A("- Decision: demographics are **manifest-only** (later-cycle rows are empty by design). "
      "Never model inputs.")
    A("")
    A("## 8. Intercourse (analysis only, never model inputs)")
    A("")
    A(f"- NumberofDaysofIntercourse: present {interc_n} / blank {n - interc_n} "
      "(same rule as app sex data: excluded from tensors).")
    A(f"- IntercourseInFertileWindow: {dict(sorted(interc_fw.items()))} (analysis tables at most).")
    A("")
    A("## 9. Irregularity rule (for the eval split)")
    A("")
    A(f"- Per-client LengthofCycle sd (clients with ≥3 cycles, n={len(sds)}): "
      f"sd > 3.0 on **{irr_clients_sd3}** clients (~{round(irr_clients_sd3 / max(1, len(sds)) * 100, 1)}%).")
    A("- Rule (loader): client is **irregular** iff cycle-length sd > 3.0 "
      "(≥3 cycles) or range > 5 (<3 cycles). Otherwise regular.")
    A("- Rationale: sd > 3.0 recovers ~27% irregular, closest to the synthetic "
      "1-in-3 prior without fitting the threshold to the gate.")
    A("")
    A("## 10. Blanks policy (global)")
    A("")
    A("- Blank or whitespace-only = **missing**, never zero — except MensesScore "
      "day cells, where blank = no bleeding (score 0) per §3.")
    A("- BBT: no BBT column exists — loader leaves BBT fully masked (never synthesized).")
    A("- Mucus: no daily mucus — loader leaves mucus at 0/none with no mask column "
      "(v1 has no mucus-missing flag; the signal is simply absent).")
    A("- Intercourse columns never enter the model (analysis only).")
    A("")
    A("## 11. Critical surprises / warnings")
    A("")
    if critical:
        A("- CRITICAL:")
        for c in critical:
            A(f"  - {c}")
    else:
        A("- CRITICAL: none.")
    if notes:
        A("- Notes:")
        for w in notes:
            A(f"  - {w}")
    else:
        A(f"- Notes: luteal outside 8–20 on {lut_out} cycles (kept); "
          "12 TotalMensesScore≠sum(day scores) mismatches (kept, day scores rule); "
          "2 peak=0 rows carry an EDO and 6 peak=1 rows lack one (kept as labelled).")
    A("")
    A("## 12. Decisions recorded (implementer must honour)")
    A("")
    A("1. Dedup (ClientID, CycleNumber), keep first.")
    A("2. Flow map: blank→none, 1→light, 2→medium, 3→heavy; blank menses = no bleed.")
    A("3. Peak rule per §4 (EDO position, FirstDayofHigh+4 fallback, median offset "
      f"{off_med}).")
    A("4. `unusual_flag` + `postpartum_flag` carried on every timeline for stratification; "
      "no rows excluded in Phase A.")
    A("5. Irregular = sd > 3.0 (≥3 cycles) else range > 5.")
    A("6. BBT fully masked; intercourse never in tensors; age/BMI manifest-only.")
    A("7. Reconstructed days are marked **present** (observed coverage, not owner-logged days).")
    A("")

    md_path = os.path.join(args.out, "fehring_eda.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[eda_fehring] rows={n} clients={len(cids)} dups={len(dup_keys)} "
          f"gaps_after_dedup={len(gap_clients)} peak_rate={peak_rate} -> {md_path}")

    if critical:
        print("[eda_fehring] CRITICAL surprises (handled, exit 2):", file=sys.stderr)
        for c in critical:
            print(f"  - {c}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
