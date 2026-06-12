"""Compatibility facade for the SPMF10 structure branch and fusion modules."""

from .dsm_structure_branch10 import DSMStructureBranch10
from .spmf10_fusion import (
    MultiScaleSPMF10,
    MultiScaleStructurePriorModulatedFusion10,
    SPMFBlock10,
    StructurePriorModulatedFusionBlock10,
)

__all__ = [
    "DSMStructureBranch10",
    "SPMFBlock10",
    "MultiScaleSPMF10",
    "StructurePriorModulatedFusionBlock10",
    "MultiScaleStructurePriorModulatedFusion10",
]
