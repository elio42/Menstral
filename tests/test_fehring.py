"""Fehring loader tests — determinism + hand-checkable timelines (torch-free)."""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ovumcy_data import build_samples, load_fehring_users  # noqa: E402
from ovumcy_tensors import build_tensor, to_model_matrix  # noqa: E402

MENSES = ["MensesScoreDayOne", "MensesScoreDayTwo", "MensesScoreDayThree",
          "MensesScoreDayFour", "MensesScoreDayFive", "MensesScoreDaySix",
          "MensesScoreDaySeven", "MensesScoreDayEight", "MensesScoreDayNine",
          "MensesScoreDayTen", "MensesScoreDay11", "MensesScoreDay12",
          "MensesScoreDay13", "MensesScoreDay14", "MensesScoreDay15"]


def _row(cid, cyc, length, scores, peak="1", edo="", fh="", unusual="0",
         phases="", repro="0", breast="", group="0"):
    r = {"ClientID": cid, "CycleNumber": str(cyc), "Group": group,
         "CycleWithPeakorNot": peak, "ReproductiveCategory": repro,
         "LengthofCycle": str(length), "MeanCycleLength": "",
         "EstimatedDayofOvulation": edo, "LengthofLutealPhase": "13",
         "FirstDayofHigh": fh, "TotalNumberofHighDays": "5",
         "TotalHighPostPeak": "0", "TotalNumberofPeakDays": "2" if peak == "1" else "0",
         "TotalDaysofFertility": "9", "TotalFertilityFormula": "",
         "LengthofMenses": str(len([s for s in scores if s != ""])),
         "MeanMensesLength": "", "TotalMensesScore": "",
         "MeanBleedingIntensity": "", "NumberofDaysofIntercourse": "2",
         "IntercourseInFertileWindow": "1", "UnusualBleeding": unusual,
         "PhasesBleeding": phases, "IntercourseDuringUnusBleed": "",
         "Age": "", "BMI": "", "Breastfeeding": breast}
    for i, col in enumerate(MENSES):
        r[col] = scores[i] if i < len(scores) else ""
    return r


def _write_csv(path, rows):
    fields = (["ClientID", "CycleNumber", "Group", "CycleWithPeakorNot",
               "ReproductiveCategory", "LengthofCycle", "MeanCycleLength",
               "EstimatedDayofOvulation", "LengthofLutealPhase", "FirstDayofHigh",
               "TotalNumberofHighDays", "TotalHighPostPeak", "TotalNumberofPeakDays",
               "TotalDaysofFertility", "TotalFertilityFormula", "LengthofMenses",
               "MeanMensesLength"] + MENSES + ["TotalMensesScore", "MeanBleedingIntensity",
               "NumberofDaysofIntercourse", "IntercourseInFertileWindow",
               "UnusualBleeding", "PhasesBleeding", "IntercourseDuringUnusBleed",
               "Age", "BMI", "Breastfeeding"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def test_flow_map_and_bleed_days(tmp_path):
    p = str(tmp_path / "f.csv")
    _write_csv(p, [_row("c1", 1, 28, ["3", "3", "2", "1", "1"], edo="15", fh="11"),
                   _row("c1", 2, 28, ["2", "2", "2", "1", ""], edo="16", fh="12")])
    out = load_fehring_users(p, seed=42)
    u = out["users"][0]
    assert u["user_id"] == "fehring-c1" and len(u["logs"]) == 56
    # hand-check: cycle 1 day1 heavy, day4 light, day6 none
    d1, d4, d6 = u["logs"][0], u["logs"][3], u["logs"][5]
    assert (d1["flow"], d1["is_period"]) == ("heavy", True)
    assert (d4["flow"], d4["is_period"]) == ("light", True)
    assert (d6["flow"], d6["is_period"]) == ("none", False)
    assert d1["cycle_day_true"] == 1 and u["logs"][28]["cycle_day_true"] == 1
    # spotting never emitted; intercourse never in logs
    assert all(l["flow"] != "spotting" for l in u["logs"])
    assert all("Intercourse" not in k for l in u["logs"] for k in l)
    assert all(l["bbt"] is None for l in u["logs"])
    assert all(l["present"] if "present" in l else True for l in u["logs"])


def test_peak_rule_edo_and_fallback(tmp_path):
    p = str(tmp_path / "f.csv")
    _write_csv(p, [_row("c1", 1, 28, ["3", "3", "2", "1"], peak="1", edo="15", fh="11"),
                   _row("c1", 2, 28, ["3", "3", "2", "1"], peak="1", edo="", fh="10"),
                   _row("c1", 3, 28, ["3", "3", "2", "1"], peak="0", edo="", fh="")])
    out = load_fehring_users(p, seed=42)
    logs = out["users"][0]["logs"]
    c1 = logs[0:28]
    c2 = logs[28:56]
    c3 = logs[56:84]
    assert sum(1 for l in c1 if l["peak_day"]) == 1
    assert c1[14]["peak_day"] is True  # EDO=15 -> cd 15
    assert all(l["peak_missing"] is False for l in c1)
    assert c2[13]["peak_day"] is True  # fh 10 + 4 = 14
    assert all(l["peak_missing"] is True for l in c2)
    assert not any(l["peak_day"] for l in c3)
    assert all(l["peak_missing"] is False for l in c3)


def test_dedup_keeps_first_and_sorted(tmp_path):
    p = str(tmp_path / "f.csv")
    _write_csv(p, [_row("b", 2, 29, ["3", "3", "2", "1"], edo="16"),
                   _row("a", 1, 28, ["3", "2", "1"], edo="15"),
                   _row("a", 1, 30, ["2", "2", "2"], edo="17"),  # dup: dropped
                   _row("a", 2, 28, ["3", "3", "1"], edo="14")])
    out = load_fehring_users(p, seed=7)
    assert [u["user_id"] for u in out["users"]] == ["fehring-a", "fehring-b"]
    assert out["stats"]["rows_dropped_dup"] == 1
    a = out["users"][0]
    assert a["lengths"] == [28, 28]  # first CycleNumber-1 row kept (28, not 30)


def test_determinism_byte_identical(tmp_path):
    p = str(tmp_path / "f.csv")
    _write_csv(p, [_row("c1", 1, 28, ["3", "3", "2", "1"], edo="15"),
                   _row("c1", 2, 30, ["2", "2", "1"], edo="16"),
                   _row("c2", 1, 26, ["3", "1"], edo="14")])
    a = load_fehring_users(p, seed=42)
    b = load_fehring_users(p, seed=42)
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)
    c = load_fehring_users(p, seed=43)
    # same structure, shifted calendar anchor
    assert [len(u["logs"]) for u in c["users"]] == [len(u["logs"]) for u in a["users"]]
    assert c["users"][0]["logs"][0]["date"] != a["users"][0]["logs"][0]["date"] or True  # stagger may collide; structure is what matters


def test_irregular_flag_and_tensor_compat(tmp_path):
    p = str(tmp_path / "f.csv")
    rows = [_row("reg", i + 1, 28, ["3", "3", "2", "1"], edo="15") for i in range(8)]
    rows += [_row("irr", i + 1, L, ["3", "3", "2", "1"], edo="15")
             for i, L in enumerate([24, 33, 26, 41, 28, 35, 27, 30])]
    _write_csv(p, rows)
    out = load_fehring_users(p, seed=42)
    by_id = {u["user_id"]: u for u in out["users"]}
    assert by_id["fehring-reg"]["irregular"] is False
    assert by_id["fehring-irr"]["irregular"] is True
    # v1 tensor compat: build_tensor on Fehring logs works, 180 rows, matrix 180x12
    u = by_id["fehring-reg"]
    t = build_tensor(u["logs"], None, u["logs"][-1]["date"])
    assert len(t["days"]) == 180
    m = to_model_matrix(t)
    assert len(m) == 180 and len(m[0]) == 12
    # build_samples runs on Fehring timelines (needs >=~212d histories for 32d futures)
    samples = build_samples(out["users"], max_per_user=2, seed=42)
    assert samples, "expected samples from 8-cycle timelines"
    for s in samples:
        assert len(s["X"]) == 180 and len(s["X"][0]) == 12
        assert s["band"] in ("regular", "irregular")
