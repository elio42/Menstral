"""PatchTST-tiny 1-2M (Track B). Torch-only; import-guarded so smoke tests run free.

Architecture: channel-independent patching (shared Conv1d over 180 daily
steps, patch_len 16 / stride 8 -> 21 patches), d_model 160, 4 heads, 4
layers, mean-pool over channels then over patches, linear quantile head to
[3,32] onset-CDF quantile curves (rows p10/p50/p90 over 32d). Target: 1-2M params.

Model I/O matches the sidecar contract: [B,180,12] -> [B,3,32].
Training loop lives in scripts/train_spike_b.py.
"""
from __future__ import annotations


def count_params(d_model=160, n_heads=4, n_layers=4, patch_len=16, n_channels=12) -> int:
    # Rough estimate: patch embed (1*d*pl) + per-layer attention (4*d^2) +
    # ffn (8*d^2) + channel-preserving head ((C*d)*96).
    per_layer = 12 * d_model * d_model
    return patch_len * d_model + per_layer * n_layers + n_channels * d_model * 96


class PatchTSTTiny:
    """Config holder; call build() to get the torch nn.Module (needs torch)."""

    def __init__(self, d_model=160, n_heads=4, n_layers=4, patch_len=16, stride=8,
                 dropout=0.1, context_len=180, pred_horizon=32, n_channels=12):
        self.cfg = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                        patch_len=patch_len, stride=stride, dropout=dropout,
                        context_len=context_len, pred_horizon=pred_horizon,
                        n_channels=n_channels)

    def build(self):
        import torch
        import torch.nn as nn

        cfg = self.cfg
        d, nl, pl = cfg["d_model"], cfg["n_layers"], cfg["patch_len"]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = cfg
                # Channel-independent patch embed: shared weights over [1,T].
                self.patch = nn.Conv1d(1, d, kernel_size=pl, stride=cfg["stride"])
                layer = nn.TransformerEncoderLayer(d_model=d, nhead=cfg["n_heads"],
                                                    dim_feedforward=4 * d, dropout=cfg["dropout"],
                                                    batch_first=True)
                self.enc = nn.TransformerEncoder(layer, num_layers=nl)
                # Channel-preserving head: mean over patches per channel, then
                # concat (NOT mean over channels — that made every sample map
                # to the same pooled vector and the net learned one constant
                # marginal CDF for all inputs).
                self.head = nn.Sequential(nn.LayerNorm(cfg["n_channels"] * d),
                                          nn.Linear(cfg["n_channels"] * d,
                                                    cfg["pred_horizon"] * 3))

            def forward(self, x):
                # x: [B,T,C] -> [B,C,T] -> fold channels -> [B*C,1,T]
                xt = x.transpose(1, 2)
                b, c, t = xt.shape
                xb = xt.reshape(b * c, 1, t)
                pr = self.patch(xb)              # [B*C,d,P]
                pr = pr.transpose(1, 2)          # [B*C,P,d]
                pr = pr.reshape(b, c, -1, d)
                h = self.enc(pr.reshape(b * c, -1, d))  # [B*C,P,d]
                h = h.reshape(b, c, -1, d).mean(dim=2)  # [B,C,d] pool patches
                q = self.head(h.reshape(b, c * d))      # [B,3*H]
                q = q.reshape(b, 3, cfg["pred_horizon"])
                return torch.sigmoid(q)          # onset-CDF values in [0,1]

        return Net()
