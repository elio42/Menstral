"""Pooled supervised dataset: 180d tensor -> 32d onset-CDF target.

Each sample: cutoff day c with a full 180d window ending at c and a full 32d
of future ground truth. X = to_model_matrix(tensor ending at c).
Y = [32] cumulative onset indicator: y[h] = 1 iff the next onset falls on a
day <= cutoff+h+1 (monotone 0..0,1..1 step; all zeros when no onset occurs
within the horizon = censored). Plus days_to_onset scalar for the eval gate.

Why cumulative, not a per-day bleed mask: the bleed mask is ~15% positive, so
pinball loss drives every quantile curve under any fixed crossing threshold
and the decoder censors everything (observed: val MAE frozen at ~31 = max
miss while train loss still fell). The CDF crosses 0.5 by construction, so
the p50 crossing estimates the median onset day and censored samples stay
censored. The shared decoder (first crossing >= 0.5) is unchanged.

Deterministic under seed. Pure numpy; torch conversion happens in the
training scripts so smoke paths stay torch-free.
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

from ovumcy_tensors import (
    build_tensor,
    is_spotting_day,
    normalize_flow,
    parse_day,
    to_model_matrix,
)
from ovumcy_tensors.tensor_v2 import (
    build_tensor_v2,
    to_model_matrix_v2,
)

HORIZON = 32


def _bleed_of(log_by_day: dict[str, dict], d: date) -> int:
    log = log_by_day.get(d.isoformat())
    if not log or not log.get("is_period"):
        return 0
    # Go bleed rule: non-spotting period day (flow or symptom reading).
    flow = normalize_flow(log.get("flow"))
    if flow == "spotting":
        return 0
    if log.get("has_spotting_symptom") and flow in ("none", ""):
        return 0
    return 1


def build_samples(
    users: list[dict],
    horizon: int = HORIZON,
    max_per_user: int = 8,
    min_history_cycles: int = 1,
    seed: int = 42,
    smooth_w: int = 2,
    tensor_version: str = "v1",
    skip_postpartum_cutoffs: bool = False,
) -> list[dict]:
    """Slide cutoffs across each user's history; deterministic.

    tensor_version: "v1" (12 cols, default — backwards compatible) or "v2"
      (14 cols with peak/peak_missing; logs without peak keys count as
      peak-unknown). skip_postpartum_cutoffs: drop cutoffs whose day carries
      the loader's postpartum_flag (Phase B training excludes postpartum
      cycles; timelines stay intact — only sampling skips them).
    """
    rng = random.Random(seed)
    samples: list[dict] = []
    use_v2 = tensor_version == "v2"
    for u in users:
        logs = u.get("logs") or []
        if not logs:
            continue
        by_day = {parse_day(l.get("date")).isoformat(): l for l in logs
                  if parse_day(l.get("date")) is not None}
        days = sorted(by_day)
        first, last = days[0], days[-1]
        # Candidate cutoffs: every 7th day keeps pooled sets small and covers
        # follicular/luteal/bleed phases without near-duplicate windows.
        cands = days[::7]
        rng.shuffle(cands)
        made = 0
        for c in cands:
            if made >= max_per_user:
                break
            if skip_postpartum_cutoffs and by_day.get(c, {}).get("postpartum_flag"):
                continue
            c_d = parse_day(c)
            # Need a full 180d window ending at c (tensor pads absent rows, so
            # strictly we need only c in range; keep the light requirement).
            # Need future ground truth: onset within horizon AND full horizon
            # of future days present in logs.
            fut_days = [(c_d + timedelta(days=h)).isoformat() for h in range(1, horizon + 1)]
            if any(f > last for f in fut_days):
                continue
            bleed = [_bleed_of(by_day, c_d + timedelta(days=h)) for h in range(1, horizon + 1)]
            if sum(bleed) == 0:
                # No onset in the next 32d (mid long-cycle): keep some as
                # negatives so the model learns "no onset soon", but cap them.
                if rng.random() < 0.7:
                    continue
                dto = horizon + 1  # censored: beyond horizon
                y = [0.0] * horizon
            else:
                dto = bleed.index(1) + 1  # first bleed day (1-based)
                # Soft step (ramp over +-smooth_w days): identical p50 crossing
                # (still 0.5 at dto) but the quantile head learns spread curves
                # instead of collapsing q10/q50/q90 onto one sharp step, which
                # previously produced zero-width intervals (coverage ~0.2).
                y = [min(1.0, max(0.0, ((h + 1) - (dto - smooth_w)) / (2 * smooth_w)))
                     for h in range(horizon)]
            # History gate: >=1 bleed day on/before c (else nothing to learn from).
            past = [d for d in days if d <= c]
            if not any(_bleed_of(by_day, parse_day(d)) for d in past[-90:]):
                continue
            hist_logs = [by_day[d] for d in past]
            if use_v2:
                tensor = build_tensor_v2(hist_logs, None, c)
                X = to_model_matrix_v2(tensor)
            else:
                tensor = build_tensor(hist_logs, None, c)
                X = to_model_matrix(tensor)
            # History context for the classical baseline on the same origin:
            # completed cycle lengths before cutoff + cycle_day at cutoff.
            from ovumcy_tensors import cycle_boundaries, days_between

            starts = cycle_boundaries(hist_logs, None, c_d)
            hist_lengths = [days_between(starts[i - 1], starts[i])
                            for i in range(1, len(starts))]
            rows = tensor["days"]
            cd_now = 0
            for r in reversed(rows):
                if r.get("present"):
                    cd_now = int(r.get("cycle_day") or 0)
                    break
            samples.append({
                "user_id": u.get("user_id"),
                "cutoff": c,
                "X": X,
                "y": [float(v) for v in y],
                "days_to_onset": int(dto),
                "censored": bool(sum(y) == 0),
                "band": ("irregular" if u.get("irregular") else "regular"),
                "hist_lengths": hist_lengths,
                "cycle_day_at_cutoff": cd_now,
                "tensor_version": ("v2" if use_v2 else "v1"),
            })
            made += 1
    # Deterministic order regardless of shuffle above.
    samples.sort(key=lambda s: (str(s["user_id"]), s["cutoff"]))
    return samples


# --- Augmentation (numpy, applied per-sample in the training loop) ---

def augment_matrix(X: list[list[float]], rng: random.Random,
                   jitter: float = 0.05, mask_p: float = 0.10) -> list[list[float]]:
    """Jitter continuous cols + random day masking (absent-row simulation).

    Width-generic: v1 [180,12] and v2 [180,14]. Peak cols (12,13) are binary
    flags and are never jittered; a masked day sets peak=0, peak_missing=1
    (an unlogged day carries no peak information).
    """
    import math

    out: list[list[float]] = []
    for row in X:
        r = list(row)
        # Jitter bbt_z (col 2), flow_norm (1), mucus_norm (4), symptom_log (5).
        for col in (1, 2, 4, 5):
            r[col] = r[col] + rng.gauss(0.0, jitter)
        if rng.random() < mask_p:
            # Simulate an unlogged day: present=0, absent=1, zero the signals,
            # keep gap/cycle cols (the tensor builder would have set them).
            r[0] = 0.0   # bleed
            r[1] = 0.0   # flow_norm
            r[2] = 0.0   # bbt_z
            r[3] = 1.0   # bbt_missing
            r[4] = 0.0
            r[5] = 0.0
            r[10] = 0.0  # present
            r[11] = 1.0  # absent
            if len(r) > 13:
                r[12] = 0.0  # peak unknown when the day is unlogged
                r[13] = 1.0  # peak_missing
        out.append(r)
    return out


def tsmix(Xa: list[list[float]], Xb: list[list[float]],
          rng: random.Random) -> tuple[list[list[float]], float]:
    """Time-series mixup: splice a suffix of sample b into sample a.

    Returns (X_mixed, lam); caller computes
    loss = lam*loss(X_mixed, ya) + (1-lam)*loss(X_mixed, yb).
    Cut point stays >=30d from either edge so the 180d phase structure
    (bleed episodes, BBT shift) survives the splice.
    """
    lam = rng.betavariate(2.0, 2.0)
    cut = rng.randint(30, len(Xa) - 30)
    mixed = [list(r) for r in Xa[:cut]] + [list(r) for r in Xb[cut:]]
    return mixed, lam
