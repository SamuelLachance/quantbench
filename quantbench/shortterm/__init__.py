"""Signaux court terme : mean-reversion + detection de regimes."""

from .signals import zscore_meanrev, ou_half_life
from .regime import detect_regimes

__all__ = ["zscore_meanrev", "ou_half_life", "detect_regimes"]
