"""Mixed training pool: synthetic + Fehring timelines (Phase B).

Deterministic under seed. Synthetic users keep integer user_ids;
Fehring users keep "fehring-<ClientID>" ids (no collisions). Each user dict
is tagged with "source" ("synthetic" | "fehring") for reporting; the
80/20 by-client split happens downstream (never by cycle — no client in
both splits).

Postpartum handling lives in sampling, not here: timelines stay intact and
`build_samples(..., skip_postpartum_cutoffs=True)` drops postpartum cutoffs
from the *training* split only. The val split keeps everything so the gate
describes deployment reality.
"""
from __future__ import annotations

from ovumcy_data.fehring import load_fehring_users
from ovumcy_data.synthetic import generate_corpus


def load_mixed_users(fehring_csv: str, n_synthetic_users: int = 400,
                     cycles_per_user: int = 8, seed: int = 42) -> dict:
    """Synthetic + Fehring users with source tags. Deterministic under seed."""
    synth = generate_corpus(n_users=n_synthetic_users,
                            cycles_per_user=cycles_per_user, seed=seed)
    fehr = load_fehring_users(fehring_csv, seed=seed)
    users: list[dict] = []
    for u in synth["users"]:
        users.append({**u, "source": "synthetic"})
    for u in fehr["users"]:
        users.append({**u, "source": "fehring"})
    return {
        "users": users,
        "stats": {
            "seed": seed,
            "n_synthetic_users": len(synth["users"]),
            "n_fehring_users": len(fehr["users"]),
            "fehring": fehr["stats"],
        },
    }
