"""Compatibility facade for DSMStructureBranch12 with the unchanged SPMF20 fusion modules."""

from .dsm_structure_branch12 import DSMStructureBranch12
from .spmf20_fusion import (
    MultiScaleStructurePriorModulatedFusion20,
    SPMFBlock20,
    StructurePriorModulatedFusionBlock20,
)

SPMFBlock21 = SPMFBlock20
MultiScaleSPMF21 = MultiScaleStructurePriorModulatedFusion20
StructurePriorModulatedFusionBlock21 = StructurePriorModulatedFusionBlock20
MultiScaleStructurePriorModulatedFusion21 = MultiScaleStructurePriorModulatedFusion20

__all__ = [
    "DSMStructureBranch12",
    "SPMFBlock21",
    "MultiScaleSPMF21",
    "StructurePriorModulatedFusionBlock21",
    "MultiScaleStructurePriorModulatedFusion21",
]
