"""Rolling-origin backtest stratified regular/irregular + FIGO bands.

For each user with >=3 cycles: hold out the last k cycles one at a time,
predict from history, score the error. Stratifies by regularity so the
report shows where the transformer wins (and where it must widen intervals).
"""
from __future__ import annotations

from typing import Callable


def rolling_origin(
    users: list[dict],
    predict_fn: Callable[[list[int]], float],
    n_holdout: int = 2,
) -> dict:
    errors: list[float] = []
    by_band: dict[str, list[float]] = {"regular": [], "irregular": []}
    per_user: list[dict] = []
    for u in users:
        lengths = [l for l in (u.get("lengths") or []) if l and l > 0]
        if len(lengths) < 3:
            continue
        band = "irregular" if u.get("irregular") else "regular"
        u_errs: list[float] = []
        for h in range(1, min(n_holdout, len(lengths) - 1) + 1):
            hist, truth = lengths[: len(lengths) - h], lengths[len(lengths) - h]
            pred = predict_fn(hist)
            err = abs(pred - truth)
            errors.append(err)
            by_band[band].append(err)
            u_errs.append(err)
        per_user.append({"user_id": u.get("user_id"), "band": band,
                         "mae": sum(u_errs) / len(u_errs) if u_errs else 0.0,
                         "n": len(u_errs)})
    from .metrics import mae as _mae  # noqa: F401 (re-export convenience)

    def _m(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n_points": len(errors),
        "n_users": len(per_user),
        "mae": _m(errors),
        "mae_regular": _m(by_band["regular"]),
        "mae_irregular": _m(by_band["irregular"]),
        "per_user": per_user,
        "errors": errors,
    }
