"""Compatibility facade for the SPMF30 fusion modules."""

from .spmf30_fusion import (
    MultiScaleSPMFFusion30,
    SPMFFusionBlock30,
)

__all__ = [
    "SPMFFusionBlock30",
    "MultiScaleSPMFFusion30",
]
