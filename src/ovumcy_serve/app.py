"""FastAPI sidecar: POST /v1/predict + GET /healthz.

Contract: mirrors Go predictor_client.go strictly.
Request:  {user_id, tensor{version,days[...]}, history_n, recent_lengths[, model_version]}
Response: {next_onset, ovulation_date, fertile_start, fertile_end,
           p10, p50, p90, model_version}  (dates YYYY-MM-DD, "" when absent)
          or {no_prediction:true, reason}.
Unknown request fields are IGNORED (forward-compatible); unknown response
fields are NEVER emitted (Go uses DisallowUnknownFields and would fallback).

Auth: shared bearer token (PREDICTOR_TOKEN), no TLS on shared bridge.
Model: artifacts/model.onnx (opset 17, static [1,180,C] -> [1,3,32],
C=12 on the v1 track, C=14 on the mixed-v2 track), decoded with the SAME
ovumcy_serve.decode used by the eval gate. The tensor version routes the
matrix builder (v1 -> to_model_matrix, v2 -> to_model_matrix_v2) and the
reshape width follows the loaded model, so a version/model mismatch fails
soft into the median fallback. interval_widen_days from the bundled
normalizer.json widens the decoded p10/p90 by k days (p50 untouched). When
no weights exist (pre-training) it falls back to the median heuristic that
still honours the contract, abstains on thin history, and widens intervals.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from datetime import date, timedelta
from typing import Any, Optional

try:
    from fastapi import FastAPI, Header, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except Exception:  # pragma: no cover - import guard for docs builds
    FastAPI = None  # type: ignore

MODEL_VERSION = os.environ.get("MODEL_VERSION", "ovumcy-base-v1-smoke")
TOKEN = os.environ.get("PREDICTOR_TOKEN", "")
MODEL_ONNX_CANDIDATES = (
    os.environ.get("MODEL_ONNX_PATH", ""),
    "artifacts/model.onnx",
    "artifacts/smoke/model.onnx",
)

_ort_session = None
_ort_path: Optional[str] = None

# Cache for interval_widen_days keyed by the resolved ONNX path: the bundled
# normalizer.json carries the calibration (v2 ships interval_widen_days: 3;
# the v1 normalizer has no such key and reads as 0 = no widening).
_widen_days_cache: dict[str, int] = {}


def _interval_widen_days(onnx_path: Optional[str]) -> int:
    """Calibration widening k read from the normalizer next to the model."""
    key = onnx_path or "default"
    if key in _widen_days_cache:
        return _widen_days_cache[key]
    widen = 0
    try:
        candidates = []
        if onnx_path:
            candidates.append(os.path.join(os.path.dirname(onnx_path), "normalizer.json"))
        candidates.append("artifacts/normalizer.json")
        for path in candidates:
            if path and os.path.exists(path):
                with open(path) as handle:
                    widen = int(json.load(handle).get("interval_widen_days") or 0)
                break
    except Exception:
        widen = 0
    if widen < 0:
        widen = 0
    _widen_days_cache[key] = widen
    return widen


def _apply_calibration(dec: dict, widen: int) -> dict:
    """Widen the decoded p10/p90 by k days each; p50 stays untouched."""
    if not widen or widen <= 0:
        return dec
    try:
        out = dict(dec)
        shifts = (("p10", -widen), ("p90", widen))
        for field, delta in shifts:
            raw = str(out.get(field) or "")
            anchor = date(int(raw[0:4]), int(raw[5:7]), int(raw[8:10]))
            out[field] = (anchor + timedelta(days=delta)).isoformat()
        return out
    except Exception:
        return dec


def _ort() -> tuple[Any, Optional[str]]:
    """Lazily load the ONNX session; returns (session|None, path|None)."""
    global _ort_session, _ort_path
    if _ort_session is not None or _ort_path == "missing":
        return (_ort_session, None if _ort_path == "missing" else _ort_path)
    for cand in MODEL_ONNX_CANDIDATES:
        if cand and os.path.exists(cand):
            try:
                import onnxruntime as ort
                _ort_session = ort.InferenceSession(cand, providers=["CPUExecutionProvider"])
                _ort_path = cand
                return _ort_session, cand
            except Exception:
                continue
    _ort_path = "missing"
    return None, None


def _decode_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))  # src/ovumcy_serve -> repo root
    for cand in (os.path.join(root, "src"), here):
        if cand not in __import__("sys").path:
            __import__("sys").path.insert(0, cand)
    return root


if FastAPI is not None:
    app = FastAPI(title="ovumcy-predictor", version="0.1.0")

    class PredictRequest(BaseModel):
        user_id: Any
        tensor: dict
        history_n: int = 0
        recent_lengths: list[int] = []
        model_version: Optional[str] = None

        class Config:
            extra = "ignore"

    _cache: dict[str, tuple[dict, float]] = {}
    CACHE_TTL = 24 * 3600
    CACHE_MAX = 1024

    def _token_ok(auth: str | None) -> bool:
        if not TOKEN:
            return True  # local smoke without token; compose always sets it
        if not auth or not auth.startswith("Bearer "):
            return False
        return auth[len("Bearer "):] == TOKEN

    def _input_hash(body: dict) -> str:
        raw = json.dumps(body.get("tensor", {}), separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(raw).hexdigest()

    def _median_next_onset(days: list[dict], recent: list[int], history_n: int) -> dict | None:
        # Find last bleed day; predict last + median(recent<=6) else 28.
        last_bleed: str | None = None
        for r in reversed(days):
            if r.get("present") and r.get("bleed"):
                last_bleed = r.get("date")
                break
        if last_bleed is None:
            # No bleed in window: use window end as anchor.
            last_bleed = days[-1]["date"] if days else date.today().isoformat()
        # Cycle-day anchor: last row's cycle_day if known else unknown.
        last_cd = 0
        for r in reversed(days):
            if r.get("present"):
                last_cd = int(r.get("cycle_day") or 0)
                break
        rec = [l for l in (recent or []) if l and l > 0][-6:]
        med = int(statistics.median(rec)) if rec else 28
        try:
            y, m, d = int(last_bleed[0:4]), int(last_bleed[5:7]), int(last_bleed[8:10])
            anchor = date(y, m, d)
        except Exception:
            return None
        # Thin history: abstain (app renders existing empty state).
        if (history_n or 0) < 1 and not rec:
            return None
        # Next onset = last boundary + med? We don't know last boundary exactly
        # from the tensor alone, so estimate: last_bleed's cycle_day -> remaining.
        if last_cd > 0:
            remaining = max(med - last_cd + 1, 1)
        else:
            remaining = med
        nxt = anchor + timedelta(days=remaining)
        ov = nxt - timedelta(days=14)
        p10 = nxt - timedelta(days=3)
        p90 = nxt + timedelta(days=3)
        return {
            "next_onset": nxt.isoformat(),
            "ovulation_date": ov.isoformat(),
            "fertile_start": (ov - timedelta(days=5)).isoformat(),
            "fertile_end": ov.isoformat(),
            "p10": p10.isoformat(), "p50": nxt.isoformat(), "p90": p90.isoformat(),
        }

    def _try_onnx(tensor: dict) -> dict | None:
        """Neural path: onnxruntime inference + shared gate decoder.

        The tensor version routes the matrix builder (v1 -> 12 cols, v2 ->
        14 cols with the libido-peak columns); the reshape width follows the
        loaded model so a version/model mismatch fails soft. Returns the
        date mapping or None (no weights / censored / error -> caller falls
        back to the median heuristic).
        """
        version = tensor.get("version") or "v1"
        days = tensor.get("days") or []
        sess, onnx_path = _ort()
        if sess is None:
            return None
        try:
            _decode_path()
            try:
                from ovumcy_serve.decode import decode_curves
                from ovumcy_tensors import to_model_matrix, to_model_matrix_v2
            except ImportError:
                from src.ovumcy_serve.decode import decode_curves  # type: ignore
                from src.ovumcy_tensors import (  # type: ignore
                    to_model_matrix,
                    to_model_matrix_v2,
                )
            import numpy as np

            if version == "v2":
                mat = to_model_matrix_v2({"version": "v2", "days": days})
            else:
                mat = to_model_matrix({"version": "v1", "days": days})
            try:
                width = int(sess.get_inputs()[0].shape[2])
            except Exception:
                width = 14 if version == "v2" else 12
            if not mat or len(mat[0]) != width:
                return None  # version/model mismatch: median fallback
            x = np.array(mat, dtype="float32").reshape(1, 180, width)
            (curves,) = sess.run(None, {"input": x})  # [1,3,32]
            cutoff = days[-1]["date"] if days else date.today().isoformat()
            curves = np.asarray(curves)
            if curves.ndim == 3 and curves.shape[0] == 1:
                curves = curves[0]  # strip batch dim -> [3,32] for the decoder
            dec = decode_curves([list(curves[0]), list(curves[1]), list(curves[2])],
                                cutoff)
            if dec is None:
                return None
            dec = _apply_calibration(dec, _interval_widen_days(onnx_path))
            return {
                "next_onset": dec["next_onset"],
                "ovulation_date": dec["ovulation_date"],
                "fertile_start": dec["fertile_start"],
                "fertile_end": dec["fertile_end"],
                "p10": dec["p10"], "p50": dec["p50"], "p90": dec["p90"],
            }
        except Exception:
            return None

    @app.get("/healthz")
    def healthz():
        sess, path = _ort()
        return {"ok": True, "model_version": MODEL_VERSION,
                "neural": bool(sess is not None),
                "onnx": path or "none"}

    @app.post("/v1/predict")
    async def predict(req: Request, authorization: str | None = Header(default=None)):
        if not _token_ok(authorization):
            return JSONResponse({"error": "unauthorized"}, status_code=401,
                                headers={"model_version": MODEL_VERSION})
        try:
            body = await req.json()
        except Exception:
            return JSONResponse({"error": "bad json"}, status_code=400,
                                headers={"model_version": MODEL_VERSION})
        tensor = body.get("tensor") or {}
        days = tensor.get("days") or []
        if tensor.get("version") not in ("v1", "v2") or len(days) != 180:
            return JSONResponse({"error": "bad tensor: want v1/v2 x180"}, status_code=422,
                                headers={"model_version": MODEL_VERSION})
        history_n = int(body.get("history_n") or 0)
        recent = list(body.get("recent_lengths") or [])
        key = f"{body.get('user_id')}:{_input_hash(body)}"
        now = time.time()
        hit = _cache.get(key)
        if hit and now - hit[1] < CACHE_TTL:
            out = dict(hit[0])
            return JSONResponse(out, headers={"model_version": out.get("model_version", MODEL_VERSION)})
        neural = _try_onnx(tensor)
        heu = neural or _median_next_onset(days, recent, history_n)
        if heu is None:
            out = {"no_prediction": True, "reason": "thin_history"}
            return JSONResponse(out, headers={"model_version": MODEL_VERSION})
        out = dict(heu)
        out["model_version"] = MODEL_VERSION
        # Strict allowlist: never emit unknown fields (Go refuses them).
        out = {k: out.get(k, "") for k in
               ["next_onset", "ovulation_date", "fertile_start", "fertile_end",
                "p10", "p50", "p90", "model_version"]}
        if len(_cache) >= CACHE_MAX:
            _cache.clear()
        _cache[key] = (out, now)
        return JSONResponse(out, headers={"model_version": out["model_version"]})

    def serve():  # pragma: no cover
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
else:
    app = None  # type: ignore
