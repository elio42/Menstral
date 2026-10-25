"""Classical baselines — the bar the transformer must beat.

Median(last<=6) is the app's current fallback (see Plan A). Mean and naive
repeat complete the ladder. No fitting, no torch, runs in ms.
"""
from __future__ import annotations

import statistics


def _recent(lengths: list[int], k: int = 6) -> list[int]:
    return [l for l in (lengths or []) if l and l > 0][-k:]


def median_predict(lengths: list[int], default: int = 28) -> float:
    r = _recent(lengths)
    return float(statistics.median(r)) if r else float(default)


def mean_predict(lengths: list[int], default: int = 28) -> float:
    r = _recent(lengths)
    return float(sum(r) / len(r)) if r else float(default)


def naive_repeat_predict(lengths: list[int], default: int = 28) -> float:
    r = _recent(lengths)
    return float(r[-1]) if r else float(default)
