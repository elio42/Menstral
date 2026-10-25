"""Smoke client: POST one synthetic tensor to the sidecar, check contract."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ovumcy_data.synthetic import generate_corpus
from ovumcy_tensors import build_tensor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--token", default=os.environ.get("PREDICTOR_TOKEN", "test-token"))
    args = p.parse_args()
    corpus = generate_corpus(n_users=1, cycles_per_user=4, seed=42)
    u = corpus["users"][0]
    tensor = build_tensor(u["logs"], None, u["logs"][-1]["date"])
    body = {"user_id": 7, "tensor": tensor, "history_n": 3,
            "recent_lengths": u["lengths"][-6:]}
    req = urllib.request.Request(
        args.url.rstrip("/") + "/v1/predict",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {args.token}"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            out = json.load(r)
            mv = r.headers.get("model_version", "")
    except Exception as e:
        raise SystemExit(f"[smoke] sidecar unreachable: {e}")
    allowed = {"next_onset", "ovulation_date", "fertile_start", "fertile_end",
               "p10", "p50", "p90", "model_version", "no_prediction", "reason"}
    extra = set(out) - allowed
    assert not extra, f"[smoke] sidecar emitted unknown fields (Go would fallback): {extra}"
    assert out.get("next_onset") or out.get("no_prediction"), f"[smoke] bad reply: {out}"
    print(f"[smoke] OK model_version={mv or out.get('model_version')} reply={out}")


if __name__ == "__main__":
    main()
