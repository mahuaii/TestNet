"""SPMF22 structure branch export.

SPMF22 combines DSMStructureBranch13 with the shared SPMF20 fusion module.
This module intentionally does not define or alias Fusion22 symbols.
"""

from .dsm_structure_branch13 import DSMStructureBranch13

__all__ = [
    "DSMStructureBranch13",
]
