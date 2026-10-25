"""Dataset + decoder + gate tests (torch-free except evaluate paths)."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ovumcy_data import augment_matrix, build_samples, generate_corpus, tsmix
from ovumcy_serve.decode import decode_curves, first_crossing
from ovumcy_eval import gate_report, median_days_to_onset
import random


def test_build_samples_shapes_and_fields():
    corpus = generate_corpus(n_users=4, cycles_per_user=6, seed=7)
    samples = build_samples(corpus["users"], horizon=32, max_per_user=4, seed=7)
    assert samples, "no samples built"
    for s in samples:
        assert len(s["X"]) == 180 and len(s["X"][0]) == 12
        assert len(s["y"]) == 32
        assert 1 <= s["days_to_onset"] <= 33
        assert "hist_lengths" in s and "cycle_day_at_cutoff" in s
        assert s["band"] in ("regular", "irregular")


def test_augment_and_tsmix_keep_shapes():
    corpus = generate_corpus(n_users=2, cycles_per_user=6, seed=11)
    samples = build_samples(corpus["users"], max_per_user=2, seed=11)
    rng = random.Random(0)
    aug = augment_matrix(samples[0]["X"], rng)
    assert len(aug) == 180 and len(aug[0]) == 12
    if len(samples) >= 2:
        mixed, lam = tsmix(samples[0]["X"], samples[1]["X"], rng)
        assert len(mixed) == 180 and 0.0 < lam < 1.0


def test_first_crossing_and_decode():
    assert first_crossing([0.1, 0.4, 0.6, 0.9]) == 3
    assert first_crossing([0.0] * 32) is None
    lo = [0.0] * 32
    mid = [0.0] * 32
    hi = [0.0] * 32
    mid[9] = 0.7   # day 10
    lo[7] = 0.8    # day 8
    hi[12] = 0.9   # day 13
    dec = decode_curves([lo, mid, hi], "2026-03-20")
    assert dec["days_to_onset_p50"] == 10
    assert dec["next_onset"] == "2026-03-30"
    assert dec["ovulation_date"] == "2026-03-16"
    flat = [[0.0] * 32, [0.0] * 32, [0.0] * 32]
    assert decode_curves(flat, "2026-03-20") is None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_gate_report_pass_fail():
    neural = [{"err": 1.0, "band": "regular"} for _ in range(12)]
    base = [3.0] * 12
    rep = gate_report(neural, base)
    assert rep["pass_mae"] and rep["delta"] == -2.0
    rep2 = gate_report([{"err": 3.0, "band": "regular"} for _ in range(12)], base)
    assert not rep2["pass"]
    assert median_days_to_onset([28, 28, 30], 11) == 18  # 28ish - 11 + 1


def test_median_baseline_on_samples():
    corpus = generate_corpus(n_users=4, cycles_per_user=6, seed=7)
    samples = build_samples(corpus["users"], max_per_user=2, seed=7)
    for s in samples:
        p = median_days_to_onset(s["hist_lengths"], s["cycle_day_at_cutoff"])
        assert 1 <= p <= 60
