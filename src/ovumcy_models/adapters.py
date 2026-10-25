"""Personal layer: gating + LoRA-r adapter budget.

Policy (Plan B, undecided — spike decides):
  n<6            frozen base, no adapter
  n=3..12        context + covariates, no weights written
  n>=12          tiny adapter (LoRA r=2-4) nightly CPU
Spike runs at n>=6 to measure personal MAE delta + CPU s/user + 2KB store.

Never a full-model copy per user. No federated learning.
"""
from __future__ import annotations


def gate_policy(n_cycles: int) -> str:
    if n_cycles >= 12:
        return "adapter"
    if n_cycles >= 3:
        return "context"
    return "frozen"


def lora_adapter_bytes(r: int = 4, d_model: int = 128, n_layers: int = 4,
                         n_adapted_layers: int = 2, bits: int = 8) -> int:
    # LoRA A (d*r) + B (r*d) per adapted proj (q+v per adapted layer).
    # Params = n_adapted * 2 projs * 2*d*r. Stored quantized (default int8).
    # Full 4-layer fp32 would be far over budget — the spike adapts only the
    # last 2 layers, which is also better for small-n (<12 cycles) stability.
    adapted = min(n_adapted_layers, n_layers)
    params = adapted * 2 * 2 * d_model * r
    return (params * bits) // 8


def adapter_fits_budget(r: int = 4, budget: int = 2048) -> bool:
    # r=2 fits 2KB int8 (2 layers); r=4 needs 4-bit packing or 4KB budget —
    # the spike measures this tradeoff (personal MAE delta vs bytes).
    if r <= 2:
        return lora_adapter_bytes(r) <= budget
    return lora_adapter_bytes(r, bits=4) <= budget
