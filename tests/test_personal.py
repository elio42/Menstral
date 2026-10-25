"""Personal gating smoke: adapter budget + policy boundaries."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ovumcy_models.adapters import adapter_fits_budget, gate_policy


def test_gating_and_budget():
    assert gate_policy(2) == "frozen"
    assert gate_policy(5) == "context"
    assert gate_policy(15) == "adapter"
    assert adapter_fits_budget(2) and adapter_fits_budget(4)
