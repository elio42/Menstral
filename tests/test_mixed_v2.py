"""Mixed v2 pipeline tests — v2 sampling, augment, mixed corpus (torch-free)."""
from __future__ import annotations

import csv
import os
import random
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ovumcy_data import augment_matrix, build_samples, load_mixed_users  # noqa: E402
from ovumcy_data.fehring import load_fehring_users  # noqa: E402
from ovumcy_tensors.tensor_v2 import to_model_matrix_v2  # noqa: E402

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


def _eight_cycles(cid, postpartum_cycle=None, **kw):
    rows = []
    for i in range(8):
        extra = {}
        if postpartum_cycle == i + 1:
            extra = {"breast": "2"}
        rows.append(_row(cid, i + 1, 28, ["3", "3", "2", "1"], edo="15",
                         fh="11", **{**kw, **extra}))
    return rows


def test_v2_peak_key_absent_is_unknown():
    # Synthetic-style logs carry no peak keys: present days are peak-unknown.
    from ovumcy_tensors import build_tensor
    from ovumcy_data.synthetic import generate_corpus
    corpus = generate_corpus(n_users=1, cycles_per_user=7, seed=5)
    u = corpus["users"][0]
    t = build_tensor(u["logs"], None, u["logs"][-1]["date"])
    m = to_model_matrix_v2({"version": "v2", "days": t["days"]})
    assert len(m[0]) == 14
    for row_m, row in zip(m, t["days"]):
        if row.get("present"):
            assert row_m[12:] == [0.0, 1.0]


def test_build_samples_v2_width_and_postpartum_skip(tmp_path):
    p = str(tmp_path / "f.csv")
    _write_csv(p, _eight_cycles("c1", postpartum_cycle=4)
               + _eight_cycles("c2"))
    users = load_fehring_users(p, seed=42)["users"]
    full = build_samples(users, max_per_user=8, seed=42, tensor_version="v2")
    assert full, "expected v2 samples from 8-cycle timelines"
    assert all(len(s["X"]) == 180 and len(s["X"][0]) == 14 for s in full)
    assert all(s["tensor_version"] == "v2" for s in full)
    skipped = build_samples(users, max_per_user=8, seed=42, tensor_version="v2",
                            skip_postpartum_cutoffs=True)
    assert 0 < len(skipped) <= len(full)
    # No kept cutoff may fall on a postpartum-flagged day.
    flag_by_day = {(u["user_id"], l["date"]) for u in users for l in u["logs"]
                   if l.get("postpartum_flag")}
    for s in skipped:
        assert (s["user_id"], s["cutoff"]) not in flag_by_day
    # v1 default path unchanged.
    v1 = build_samples(users, max_per_user=2, seed=42)
    assert all(len(s["X"][0]) == 12 for s in v1)


def test_augment_v2_mask_sets_peak_unknown():
    rng = random.Random(0)
    row14 = [0.0] * 14
    row14[10] = 1.0
    out = augment_matrix([list(row14)], rng, jitter=0.0, mask_p=1.0)
    assert out[0][10] == 0.0 and out[0][11] == 1.0
    assert out[0][12] == 0.0 and out[0][13] == 1.0
    # Without masking, binary peak cols are never jittered.
    rng = random.Random(1)
    row14[12] = 1.0
    row14[13] = 0.0
    out = augment_matrix([list(row14)], rng, jitter=0.5, mask_p=0.0)
    assert out[0][12] == 1.0 and out[0][13] == 0.0


def test_mixed_loader_tags_and_counts(tmp_path):
    p = str(tmp_path / "f.csv")
    _write_csv(p, [_row("c1", 1, 28, ["3", "2"], edo="15"),
                _row("c2", 1, 28, ["3", "2"], edo="15")])
    mix = load_mixed_users(p, n_synthetic_users=4, cycles_per_user=2, seed=42)
    assert mix["stats"]["n_synthetic_users"] == 4
    assert mix["stats"]["n_fehring_users"] == 2
    srcs = {u["user_id"]: u["source"] for u in mix["users"]}
    assert all(v == "synthetic" for k, v in srcs.items() if not str(k).startswith("fehring-"))
    assert all(v == "fehring" for k, v in srcs.items() if str(k).startswith("fehring-"))
    assert len(mix["users"]) == 6
