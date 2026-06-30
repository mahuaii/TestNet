"""Compatibility facade for the SPMF11 structure branch and fusion modules."""

from .dsm_structure_branch11 import DSMStructureBranch11
from .spmf11_fusion import (
    MultiScaleSPMFFusion11,
    SPMFFusionBlock11,
)

__all__ = [
    "DSMStructureBranch11",
    "SPMFFusionBlock11",
    "MultiScaleSPMFFusion11",
]
