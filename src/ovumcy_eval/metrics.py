"""Metrics: MAE/RMSE/MedianAE + interval coverage + Wilcoxon."""
from __future__ import annotations


def mae(y_true: list[float], y_pred: list[float]) -> float:
    assert len(y_true) == len(y_pred) and y_true
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    assert len(y_true) == len(y_pred) and y_true
    return (sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)) ** 0.5


def medae(y_true: list[float], y_pred: list[float]) -> float:
    import statistics
    errs = sorted(abs(a - b) for a, b in zip(y_true, y_pred))
    return float(statistics.median(errs))


def coverage(y_true: list[float], lo: list[float], hi: list[float]) -> float:
    assert len(y_true) == len(lo) == len(hi) and y_true
    hit = sum(1 for a, l, h in zip(y_true, lo, hi) if l <= a <= h)
    return hit / len(y_true)


def wilcoxon_p(a_errs: list[float], b_errs: list[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value on paired absolute errors.

    Uses scipy when available, else a normal-approximation fallback so the
    smoke path has no hard scipy dependency at import time.
    """
    assert len(a_errs) == len(b_errs) and a_errs
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(a_errs, b_errs, zero_method="wilcox", alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        diffs = [a - b for a, b in zip(a_errs, b_errs) if a != b]
        if not diffs:
            return 1.0
        import math
        n = len(diffs)
        ranked = sorted(range(n), key=lambda i: abs(diffs[i]))
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and abs(diffs[ranked[j + 1]]) == abs(diffs[ranked[i]]):
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[ranked[k]] = avg
            i = j + 1
        w = sum(r for d, r in zip(diffs, ranks) if d > 0)
        mu = n * (n + 1) / 4.0
        var = n * (n + 1) * (2 * n + 1) / 24.0
        if var <= 0:
            return 1.0
        z = (w - mu) / math.sqrt(var)
        # two-sided normal tail
        return float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))))
