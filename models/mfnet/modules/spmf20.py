"""Compatibility facade for the shared structure branch and SPMF20 fusion modules."""

from .dsm_structure_branch10 import DSMStructureBranch10
from .spmf20_fusion import (
    MultiScaleSPMFFusion20,
    SPMFFusionBlock20,
)

__all__ = [
    "DSMStructureBranch10",
    "SPMFFusionBlock20",
    "MultiScaleSPMFFusion20",
]
