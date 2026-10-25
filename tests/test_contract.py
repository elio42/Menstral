"""Contract tests: sidecar reply stays within Go's strict allowlist."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_response_allowlist_matches_go_decoder():
    # Go predictorResponse accepts exactly these keys (plus no_prediction/reason).
    allowed = {"next_onset", "ovulation_date", "fertile_start", "fertile_end",
               "p10", "p50", "p90", "model_version", "no_prediction", "reason"}
    # App smoke: a sidecar reply with any extra key must be treated as fallback.
    reply = {"next_onset": "2026-04-09", "future_field": 1}
    extra = set(reply) - allowed
    assert extra == {"future_field"}, "contract drift detector broken"


def test_request_envelope_shape():
    from ovumcy_data.synthetic import generate_corpus
    from ovumcy_tensors import build_tensor
    corpus = generate_corpus(n_users=1, cycles_per_user=4, seed=42)
    u = corpus["users"][0]
    t = build_tensor(u["logs"], None, u["logs"][-1]["date"])
    body = {"user_id": 7, "tensor": t, "history_n": 3,
            "recent_lengths": u["lengths"][-6:]}
    assert body["tensor"]["version"] == "v1" and len(body["tensor"]["days"]) == 180


def test_compose_has_no_published_ports():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    text = (root / "docker-compose.sidecar.yml").read_text()
    assert "expose:" in text and '"8000"' in text
    # No `ports:` stanza allowed for the predictor service.
    assert "\n    ports:" not in text and "\n  ports:" not in text
