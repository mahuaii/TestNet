"""Compatibility facade for the SPMF11 structure branch and fusion modules."""

from .dsm_structure_branch11 import DSMStructureBranch11
from .spmf11_fusion import (
    MultiScaleSPMF11,
    MultiScaleStructurePriorModulatedFusion11,
    SPMFBlock11,
    StructurePriorModulatedFusionBlock11,
)

__all__ = [
    "DSMStructureBranch11",
    "SPMFBlock11",
    "MultiScaleSPMF11",
    "StructurePriorModulatedFusionBlock11",
    "MultiScaleStructurePriorModulatedFusion11",
]
