"""Compatibility facade for DSMStructureBranch13 with the unchanged SPMF20 fusion modules."""

from .dsm_structure_branch13 import DSMStructureBranch13
from .spmf20_fusion import (
    MultiScaleStructurePriorModulatedFusion20,
    SPMFBlock20,
    StructurePriorModulatedFusionBlock20,
)

SPMFBlock22 = SPMFBlock20
MultiScaleSPMF22 = MultiScaleStructurePriorModulatedFusion20
StructurePriorModulatedFusionBlock22 = StructurePriorModulatedFusionBlock20
MultiScaleStructurePriorModulatedFusion22 = MultiScaleStructurePriorModulatedFusion20

__all__ = [
    "DSMStructureBranch13",
    "SPMFBlock22",
    "MultiScaleSPMF22",
    "StructurePriorModulatedFusionBlock22",
    "MultiScaleStructurePriorModulatedFusion22",
]
