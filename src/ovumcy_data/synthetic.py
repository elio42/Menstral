"""MIT synthetic daily-log generator.

Produces plausible daily rows for pooled pre-training. NOT a physiological
simulator — just enough structure that baselines and spikes have something to
beat / learn: regular + irregular users, FIGO-like bands, biphasic BBT with
Zhu-like retrospectivity, mucus peak, sparse/missing BBT.

Deterministic under seed. No network, no raw-data bundling.
"""
from __future__ import annotations

import math
import random
from datetime import date, timedelta
from typing import Any


def _bbt(cycle_day: int, ov_day: int, rng: random.Random, disturbed: bool) -> float | None:
    if rng.random() < 0.35:  # sparse BBT: often unmeasured
        return None
    base = 36.35 if cycle_day <= ov_day else 36.85
    v = rng.gauss(base, 0.15)
    if disturbed:
        v += rng.gauss(0.3, 0.2)
    return round(v, 2)


def generate_user(
    user_id: int,
    start: date,
    n_cycles: int,
    rng: random.Random,
    irregular: bool = False,
) -> tuple[list[dict], list[int]]:
    logs: list[dict] = []
    lengths: list[int] = []
    day = start
    for _ in range(n_cycles):
        length = rng.randint(24, 34) if not irregular else rng.choice(
            [22, 24, 28, 33, 41, 55, 68])
        lengths.append(length)
        ov_day = length - 14 + rng.randint(-2, 2)
        period_len = rng.randint(3, 6)
        for cd in range(1, length + 1):
            is_period = cd <= period_len
            flow = "none"
            if is_period:
                flow = rng.choice(["light", "medium", "medium", "heavy"]) if cd <= 2 else rng.choice(
                    ["spotting", "light", "light", "medium"])
                if cd == period_len and rng.random() < 0.3:
                    flow = "spotting"
            mucus = "none"
            if not is_period:
                if ov_day - 5 <= cd <= ov_day:
                    mucus = rng.choice(["moist", "creamy", "eggwhite", "eggwhite"])
                elif cd < ov_day:
                    mucus = rng.choice(["none", "none", "dry", "moist"])
                else:
                    mucus = rng.choice(["none", "none", "dry"])
            disturbed = rng.random() < 0.05
            factors = []
            if disturbed:
                factors.append(rng.choice(["illness", "sleep_disruption"]))
            bbt = _bbt(cd, ov_day, rng, disturbed)
            sym: list[int] = []
            if rng.random() < 0.25:
                sym = rng.sample([1, 2, 3, 4, 5, 6], k=rng.randint(1, 3))
            logs.append({
                "date": day.isoformat(),
                "is_period": is_period,
                "flow": flow,
                "bbt": bbt,
                "cervical_mucus": mucus,
                "symptom_ids": sym,
                "cycle_factor_keys": factors,
                "cycle_start": (cd == 1 and is_period and flow != "spotting"),
                "is_uncertain": False,
                "has_spotting_symptom": False,
            })
            day += timedelta(days=1)
    return logs, lengths


def generate_corpus(
    n_users: int = 12,
    cycles_per_user: int = 4,
    seed: int = 42,
    start: date | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    start = start or date(2024, 6, 1)
    users = []
    for u in range(n_users):
        irregular = (u % 3 == 2)  # every third user irregular (FIGO-like band)
        logs, lengths = generate_user(u, start + timedelta(days=rng.randint(0, 30)),
                                      cycles_per_user, rng, irregular)
        users.append({"user_id": u, "irregular": irregular, "logs": logs, "lengths": lengths})
    return {"seed": seed, "users": users}


def fehring_like_user(user_id: int, seed: int = 42) -> dict:
    """Fehring/Marquette-LIKE synthetic BBT+mucus (no raw data bundled).

    The real Fehring 1665-cycle set is research-OK but must not be bundled
    without permission. This generator reproduces its *shape* (daily BBT +
    mucus score over ~6 cycles) so pipelines run offline. To use the real
    set, place the licensed CSV at data/fehring_raw.csv and run
    scripts/build_tensors.py --fehring data/fehring_raw.csv (never commit it).
    """
    rng = random.Random(seed + user_id)
    logs, lengths = generate_user(user_id, date(2023, 1, 2), 6, rng, irregular=(user_id % 2 == 0))
    return {"user_id": f"fehring-like-{user_id}", "logs": logs, "lengths": lengths}
