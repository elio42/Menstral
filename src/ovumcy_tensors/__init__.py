"""Package init: tensor v1 public surface (+ v2 builder)."""
from .tensor_v1 import (
    FEATURE_COLUMNS,
    MASK_ABSENT_ROW,
    MASK_BBT,
    MASK_CYCLE_DAY,
    NUM_FEATURES,
    SKIP_GAP_DAYS,
    VERSION,
    WINDOW_DAYS,
    build_tensor,
    cycle_boundaries,
    days_between,
    flow_score,
    hash_tensor,
    is_spotting_day,
    mucus_score,
    normalize_flow,
    parse_day,
    to_model_matrix,
)
from .tensor_v2 import (
    build_tensor_v2,
    to_model_matrix_v2,
)

__all__ = [
    "FEATURE_COLUMNS", "MASK_ABSENT_ROW", "MASK_BBT", "MASK_CYCLE_DAY",
    "NUM_FEATURES", "SKIP_GAP_DAYS", "VERSION", "WINDOW_DAYS",
    "build_tensor", "cycle_boundaries", "days_between", "flow_score",
    "hash_tensor", "is_spotting_day", "mucus_score", "normalize_flow",
    "parse_day", "to_model_matrix",
    "build_tensor_v2", "to_model_matrix_v2",
]
