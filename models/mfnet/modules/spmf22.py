"""Compatibility facade for DSMStructureBranch13 with the unchanged SPMF20 fusion modules."""

from .dsm_structure_branch13 import DSMStructureBranch13
from .spmf20_fusion import (
    MultiScaleSPMFFusion20,
    SPMFFusionBlock20,
)

SPMFFusionBlock22 = SPMFFusionBlock20
MultiScaleSPMFFusion22 = MultiScaleSPMFFusion20

__all__ = [
    "DSMStructureBranch13",
    "SPMFFusionBlock22",
    "MultiScaleSPMFFusion22",
]
