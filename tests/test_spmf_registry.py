from __future__ import annotations

import unittest

from models.mfnet.UNetFormer_MMSAM_spmf import SPMF_VARIANTS, UNetFormerSPMF
from models.mfnet.UNetFormer_MMSAM_spmf10 import UNetFormerSPMF10
from models.mfnet.UNetFormer_MMSAM_spmf11 import UNetFormerSPMF11
from models.mfnet.UNetFormer_MMSAM_spmf20 import UNetFormerSPMF20
from models.mfnet.UNetFormer_MMSAM_spmf21 import UNetFormerSPMF21
from models.mfnet.UNetFormer_MMSAM_spmf22 import UNetFormerSPMF22
from models.mfnet.modules import (
    DSMStructureBranch10,
    DSMStructureBranch11,
    DSMStructureBranch12,
    DSMStructureBranch13,
    MultiScaleStructurePriorModulatedFusion10,
    MultiScaleStructurePriorModulatedFusion11,
    MultiScaleStructurePriorModulatedFusion20,
    MultiScaleStructurePriorModulatedFusion21,
    MultiScaleStructurePriorModulatedFusion22,
)


class SPMFVariantRegistryTest(unittest.TestCase):
    def test_public_variants_share_unified_spmf_shell(self) -> None:
        for cls, variant_name in (
            (UNetFormerSPMF10, "10"),
            (UNetFormerSPMF11, "11"),
            (UNetFormerSPMF20, "20"),
            (UNetFormerSPMF21, "21"),
            (UNetFormerSPMF22, "22"),
        ):
            self.assertTrue(issubclass(cls, UNetFormerSPMF))
            self.assertEqual(cls.spmf_variant, variant_name)

    def test_variant_specs_keep_legacy_attribute_names_and_module_combinations(self) -> None:
        expected = {
            "10": (
                DSMStructureBranch10,
                MultiScaleStructurePriorModulatedFusion10,
                "spmf10_indexes",
                "structure_branch10",
                "spmf10",
                (("spmf10", "spmf10", "spmf10"), ("structure10", "structure_branch10", "spmf10/structure")),
            ),
            "11": (
                DSMStructureBranch11,
                MultiScaleStructurePriorModulatedFusion11,
                "spmf11_indexes",
                "structure_branch11",
                "spmf11",
                (("spmf11", "spmf11", "spmf11"), ("structure11", "structure_branch11", "spmf11/structure")),
            ),
            "20": (
                DSMStructureBranch10,
                MultiScaleStructurePriorModulatedFusion20,
                "spmf20_indexes",
                "structure_branch10",
                "spmf20",
                (("spmf20", "spmf20", "spmf20"), ("structure10", "structure_branch10", "spmf20/structure")),
            ),
            "21": (
                DSMStructureBranch12,
                MultiScaleStructurePriorModulatedFusion21,
                "spmf21_indexes",
                "structure_branch12",
                "spmf21",
                (
                    ("spmf21", "spmf21", "spmf21"),
                    ("structure12", "structure_branch12", "spmf21/structure"),
                    ("structure21", "structure_branch12", "spmf21/structure"),
                ),
            ),
            "22": (
                DSMStructureBranch13,
                MultiScaleStructurePriorModulatedFusion22,
                "spmf22_indexes",
                "structure_branch13",
                "spmf22",
                (
                    ("spmf22", "spmf22", "spmf22"),
                    ("structure13", "structure_branch13", "spmf22/structure"),
                    ("structure22", "structure_branch13", "spmf22/structure"),
                ),
            ),
        }

        self.assertEqual(set(SPMF_VARIANTS), set(expected))
        for variant_name, (
            structure_branch_cls,
            fusion_cls,
            indexes_attr,
            structure_attr,
            fusion_attr,
            intermediate_modules,
        ) in expected.items():
            spec = SPMF_VARIANTS[variant_name]
            self.assertIs(spec.structure_branch_cls, structure_branch_cls)
            self.assertIs(spec.fusion_cls, fusion_cls)
            self.assertEqual(spec.indexes_attr, indexes_attr)
            self.assertEqual(spec.structure_attr, structure_attr)
            self.assertEqual(spec.fusion_attr, fusion_attr)
            self.assertEqual(spec.intermediate_modules, intermediate_modules)


if __name__ == "__main__":
    unittest.main()
