"""Compatibility facade for DSMStructureBranch12 with the unchanged SPMF20 fusion modules."""

from .dsm_structure_branch12 import DSMStructureBranch12
from .spmf20_fusion import (
    MultiScaleSPMFFusion20,
    SPMFFusionBlock20,
)

SPMFFusionBlock21 = SPMFFusionBlock20
MultiScaleSPMFFusion21 = MultiScaleSPMFFusion20

__all__ = [
    "DSMStructureBranch12",
    "SPMFFusionBlock21",
    "MultiScaleSPMFFusion21",
]
