"""AGPL-safe synthetic corpus generators (MIT) + pooled dataset + Fehring loader."""
from .dataset import HORIZON, augment_matrix, build_samples, tsmix
from .fehring import FLOW_MAP, PEAK_FALLBACK_OFFSET, load_fehring_users
from .mixed import load_mixed_users
from .synthetic import fehring_like_user, generate_corpus, generate_user

__all__ = ["generate_user", "generate_corpus", "fehring_like_user",
           "HORIZON", "build_samples", "augment_matrix", "tsmix",
           "load_fehring_users", "FLOW_MAP", "PEAK_FALLBACK_OFFSET",
           "load_mixed_users"]
