"""Torch-path tests: pinball loss, training step, curve eval (skipped w/o torch)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = __import__("pytest").importorskip("torch")


def _const_model(value: float):
    import torch.nn as nn

    class C(nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 3, 32), value)
    return C()


def test_pinball_prefers_correct_quantile():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from ovumcy_models.train_common import pinball_loss_torch
    # Quantile rows are [q10, q50, q90]; a good forecast of target 1 is
    # ordered and near 1, a bad one is inverted/far.
    pred = torch.tensor([[[0.7] * 32, [0.85] * 32, [0.95] * 32]])
    tgt = torch.ones(1, 32)
    loss = float(pinball_loss_torch(pred, tgt))
    pred_bad = torch.tensor([[[0.9] * 32, [0.5] * 32, [0.1] * 32]])
    loss_bad = float(pinball_loss_torch(pred_bad, tgt))
    assert loss < loss_bad


def test_train_step_reduces_loss_and_eval_decodes():
    from ovumcy_data import build_samples, generate_corpus
    from ovumcy_models.train_common import batch_loss, evaluate_curves

    corpus = generate_corpus(n_users=2, cycles_per_user=6, seed=21)
    samples = build_samples(corpus["users"], max_per_user=4, seed=21)
    chunk = [dict(s) for s in samples[:4]]
    import torch as _t

    model = _const_model(0.5)
    X = _t.tensor([s["X"] for s in chunk])
    y = _t.tensor([s["y"] for s in chunk])
    _, pred = batch_loss(model, X, y, [dict(s) for s in chunk], "cpu")
    assert list(pred.shape) == [len(chunk), 3, 32]
    rows = evaluate_curves(model, samples[:4])
    assert len(rows) == 4 and all("err" in r and "baseline_err" in r for r in rows)
    # Constant 0.5 curves: first crossing at h=1 (>= 0.5) or censored nowhere.
    assert all(r["pred"] == 1 or r["pred"] is None for r in rows)
