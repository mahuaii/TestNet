"""Compatibility facade for the shared structure branch and SPMF20 fusion modules."""

from .dsm_structure_branch10 import DSMStructureBranch10
from .spmf20_fusion import (
    MultiScaleSPMF20,
    MultiScaleStructurePriorModulatedFusion20,
    SPMFBlock20,
    StructurePriorModulatedFusionBlock20,
)

__all__ = [
    "DSMStructureBranch10",
    "SPMFBlock20",
    "MultiScaleSPMF20",
    "StructurePriorModulatedFusionBlock20",
    "MultiScaleStructurePriorModulatedFusion20",
]
