"""Compatibility facade for the SPMF10 structure branch and fusion modules."""

from .dsm_structure_branch10 import DSMStructureBranch10
from .spmf10_fusion import (
    MultiScaleSPMFFusion10,
    SPMFFusionBlock10,
)

__all__ = [
    "DSMStructureBranch10",
    "SPMFFusionBlock10",
    "MultiScaleSPMFFusion10",
]
