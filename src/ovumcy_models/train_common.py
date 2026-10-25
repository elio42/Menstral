"""Shared torch training utilities (both spikes + Phase B mixed v2).

Import-guarded: importing this module never imports torch; every function
imports torch locally so smoke/test paths without torch still work.
Model I/O: [B,180,C] -> [B,3,32] onset-CDF quantile curves (C=12 v1, C=14 v2).
Loss: pinball (quantile) loss against the cumulative onset indicator.
"""
from __future__ import annotations

import random

QUANTILES = (0.1, 0.5, 0.9)


def pinball_loss_torch(pred, target):
    """Mean pinball loss. pred [B,3,H] (or [3,H]), target [B,H] (or [H])."""
    import torch

    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target.dim() == 1:
        target = target.unsqueeze(0)
    qs = torch.tensor(QUANTILES, device=pred.device).view(1, 3, 1)
    y = target.unsqueeze(1).expand_as(pred)
    err = y - pred
    return torch.mean(torch.maximum(qs * err, (qs - 1.0) * err))


def make_batches(samples: list[dict], batch_size: int, rng: random.Random,
                 shuffle: bool = True):
    """Yield chunks of samples."""
    idx = list(range(len(samples)))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        yield [samples[i] for i in idx[s:s + batch_size]]


def to_tensors(chunk: list[dict], device: str = "cpu",
               augment: bool = False, rng: random.Random | None = None,
               tsmix_p: float = 0.0):
    """Chunk -> (X [B,180,C], y [B,32]). Applies jitter/mask (+tsmix marks).

    C is 12 (v1) or 14 (v2); width comes from the samples. TSmix is encoded
    as chunk[i]["_mix"] = (ya, yb, lam) consumed by batch_loss.
    """
    import torch

    from ovumcy_data.dataset import augment_matrix, tsmix

    Xs = [s["X"] for s in chunk]
    ys = [s["y"] for s in chunk]
    if augment and rng is not None:
        Xs = [augment_matrix(x, rng) for x in Xs]
        if tsmix_p > 0 and len(chunk) >= 2 and rng.random() < tsmix_p:
            i, j = rng.sample(range(len(chunk)), 2)
            mixed, lam = tsmix(Xs[i], Xs[j], rng)
            Xs[i] = mixed
            chunk[i]["_mix"] = (ys[i], ys[j], lam)
    X = torch.tensor(Xs, dtype=torch.float32, device=device)
    y = torch.tensor(ys, dtype=torch.float32, device=device)
    return X, y


def batch_loss(model, X, y, chunk, device: str = "cpu"):
    """Forward + pinball loss, honouring tsmix mixup weights."""
    import torch

    pred = model(X)
    qs = torch.tensor(QUANTILES, device=pred.device).view(1, 3, 1)
    ye = y.unsqueeze(1).expand_as(pred)
    per = torch.mean(torch.maximum(qs * (ye - pred), (qs - 1.0) * (ye - pred)),
                     dim=(1, 2))  # [B]
    total = per.sum()
    n_mix = 0
    for k, s in enumerate(chunk):
        m = s.pop("_mix", None) if isinstance(s, dict) else None
        if m is None:
            continue
        ya, yb, lam = m
        t = torch.tensor([ya, yb], dtype=torch.float32, device=device)  # [2,H]
        p = pred[k:k + 1].expand(2, -1, -1)  # [2,3,H]
        te = t.unsqueeze(1).expand_as(p)
        per_mix = torch.mean(torch.maximum(qs * (te - p), (qs - 1.0) * (te - p)),
                             dim=(1, 2))  # [2]
        total = total - per[k] + lam * per_mix[0] + (1.0 - lam) * per_mix[1]
        n_mix += 1
    return total / max(len(chunk), 1), pred


def train_one_epoch(model, samples: list[dict], optim, device: str,
                    batch_size: int, rng: random.Random,
                    augment: bool = True, tsmix_p: float = 0.3) -> float:
    """One epoch over samples; returns mean batch loss."""
    model.train()
    tot, n = 0.0, 0
    for chunk in make_batches(samples, batch_size, rng, shuffle=True):
        optim.zero_grad()
        X, y = to_tensors(chunk, device, augment=augment, rng=rng, tsmix_p=tsmix_p)
        loss, _ = batch_loss(model, X, y, chunk, device)
        loss.backward()
        optim.step()
        tot += float(loss.detach().cpu())
        n += 1
    return tot / max(n, 1)


def evaluate_curves(model, samples: list[dict], device: str = "cpu",
                    batch_size: int = 64) -> list[dict]:
    """Run inference; decode each sample with the shared serve decoder.

    Returns per-sample {pred (or None if censored), truth, err}.
    Censored model output (no p50 crossing) with non-censored truth scores
    err = horizon (max miss); censored truth + censored pred scores 0.
    """
    import torch

    from ovumcy_eval.gate import median_days_to_onset
    from ovumcy_serve.decode import HORIZON, decode_curves

    model.eval()
    out: list[dict] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            chunk = samples[start:start + batch_size]
            X = torch.tensor([s["X"] for s in chunk], dtype=torch.float32, device=device)
            pred = model(X).detach().cpu().tolist()  # [B,3,H]
            for s, curves in zip(chunk, pred):
                dec = decode_curves(curves, s["cutoff"])
                truth = int(s["days_to_onset"])
                censored_truth = bool(s.get("censored"))
                truth_c = HORIZON + 1 if censored_truth else truth
                # Classical baseline on the identical origin.
                b_pred = median_days_to_onset(s.get("hist_lengths") or [],
                                              int(s.get("cycle_day_at_cutoff") or 0))
                b_err = abs(b_pred - truth_c)
                if dec is None:
                    err = 0.0 if censored_truth else float(HORIZON)
                    out.append({"pred": None, "plo": None, "phi": None,
                                "truth": truth_c, "err": err, "baseline_err": float(b_err),
                                "band": s.get("band"), "censored": censored_truth})
                else:
                    p = int(dec["days_to_onset_p50"])
                    out.append({"pred": p, "plo": int(dec["days_to_onset_p10"]),
                                "phi": int(dec["days_to_onset_p90"]),
                                "truth": truth_c, "err": abs(p - truth_c),
                                "baseline_err": float(b_err),
                                "band": s.get("band"), "censored": censored_truth})
    return out
