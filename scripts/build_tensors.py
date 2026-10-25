"""Build tensors from synthetic (+ optional Fehring CSV) corpus."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ovumcy_data.synthetic import generate_corpus
from ovumcy_tensors import build_tensor


def main() -> None:
    p = argparse.ArgumentParser(description="Build v1 tensor corpus (smoke by default).")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default="data/smoke")
    p.add_argument("--fehring", default=None, help="licensed Fehring CSV (never committed)")
    args = p.parse_args()
    n_users = 12
    cyc = 4
    try:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        n_users = int(cfg.get("data", {}).get("smoke_users", n_users)) if args.smoke else int(
            cfg.get("data", {}).get("full_users", 400))
        cyc = int(cfg.get("data", {}).get("smoke_cycles_per_user", cyc)) if args.smoke else int(
            cfg.get("data", {}).get("full_cycles_per_user", 8))
    except Exception as e:
        print(f"[build_tensors] config fallback ({e}); using defaults")
    corpus = generate_corpus(n_users=n_users, cycles_per_user=cyc, seed=42)
    os.makedirs(args.out, exist_ok=True)
    n_rows = 0
    for u in corpus["users"]:
        logs = u["logs"]
        today = logs[-1]["date"]
        tensor = build_tensor(logs, None, today)
        assert len(tensor["days"]) == 180, "tensor must be 180 rows"
        with open(os.path.join(args.out, f"user_{u['user_id']}.json"), "w") as f:
            json.dump({"user_id": u["user_id"], "tensor": tensor, "lengths": u["lengths"]}, f)
        n_rows += 1
    manifest = {"sources": [{"name": "synthetic-mittm", "license": "MIT",
                              "url": "generated-locally", "seed": 42}],
                "n_users": n_rows, "smoke": bool(args.smoke)}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if args.fehring:
        print("[build_tensors] Fehring CSV accepted but never copied; "
              "parse it in-memory only (see synthetic.fehring_like_user).")
    print(f"[build_tensors] wrote {n_rows} tensors -> {args.out}/")


if __name__ == "__main__":
    main()
