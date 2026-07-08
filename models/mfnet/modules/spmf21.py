"""SPMF21 structure branch export.

SPMF21 combines DSMStructureBranch12 with the shared SPMF20 fusion module.
This module intentionally does not define or alias Fusion21 symbols.
"""

from .dsm_structure_branch12 import DSMStructureBranch12

__all__ = [
    "DSMStructureBranch12",
]
