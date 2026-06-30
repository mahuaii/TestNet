from __future__ import annotations

import importlib
import sys
import types
import unittest

from multimodal_helpers import _fake_mfnet_optional_imports


class MFNetBuildTest(unittest.TestCase):
    def test_build_model_passes_sam_checkpoint_to_mfnet(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormer:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormer = FakeUNetFormer
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "mfnet_unetformer",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormer)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_testnet_mmadapter10(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerMMAdapter10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerMMAdapter10 = FakeUNetFormerMMAdapter10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_mmadapter10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerMMAdapter10)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_testnet_mmadapter20(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerMMAdapter20:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerMMAdapter20 = FakeUNetFormerMMAdapter20
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_mmadapter20",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerMMAdapter20)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_testnet_mmadapter21(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerMMAdapter21:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerMMAdapter21 = FakeUNetFormerMMAdapter21
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_mmadapter21",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerMMAdapter21)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_dga(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA10 = FakeUNetFormerDGA10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_dga10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA10)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": False,
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_dga2(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA20:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA20 = FakeUNetFormerDGA20
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_dga20",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA20)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": False,
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_dga20_dgsf10(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA20DGSF10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA20DGSF10 = FakeUNetFormerDGA20DGSF10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_dga20_dgsf10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA20DGSF10)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_dgsf10(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGSF10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGSF10 = FakeUNetFormerDGSF10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_dgsf10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["dgsf10"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGSF10)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                        "record_intermediate_modules": ["dgsf10"],
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_dgfm(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGFM:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGFM = FakeUNetFormerDGFM
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_dgfm",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["dgfm"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGFM)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                        "record_intermediate_modules": ["dgfm"],
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_dgfm01(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGFM01:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGFM01 = FakeUNetFormerDGFM01
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_dgfm01",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["dgfm01"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGFM01)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                        "record_intermediate_modules": ["dgfm01"],
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_dgfm01_upernet(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGFM01UperNet:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGFM01UperNet = FakeUNetFormerDGFM01UperNet
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_dgfm01_upernet",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["dgfm01_upernet"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGFM01UperNet)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                        "record_intermediate_modules": ["dgfm01_upernet"],
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_sgcf(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerSGCF:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerSGCF = FakeUNetFormerSGCF
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_sgcf",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["sgcf"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerSGCF)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                        "record_intermediate_modules": ["sgcf"],
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_spmf10(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerSPMF10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerSPMF10 = FakeUNetFormerSPMF10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_spmf10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["spmf_fusion10"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerSPMF10)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                        "record_intermediate_modules": ["spmf_fusion10"],
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_spmf20(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerSPMF20:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerSPMF20 = FakeUNetFormerSPMF20
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_spmf20",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["spmf_fusion20"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerSPMF20)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                        "record_intermediate_modules": ["spmf_fusion20"],
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign_spmf20(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignSPMF20:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignSPMF20 = FakeUNetFormerPreAlignSPMF20
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_prealign_spmf20",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["spmf_fusion20"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignSPMF20)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                        "record_intermediate_modules": ["spmf_fusion20"],
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_mfnet_exports_unetformer_prealign_spmf20(self) -> None:
        with _fake_mfnet_optional_imports():
            mfnet_module = importlib.import_module("models.mfnet")
            exported_cls = getattr(mfnet_module, "UNetFormerPreAlignSPMF20")
            from models.mfnet.UNetFormer_MMSAM_prealign_spmf20 import UNetFormerPreAlignSPMF20

        self.assertIs(exported_cls, UNetFormerPreAlignSPMF20)

    def test_build_model_dispatches_to_mfnet_prealign_auxalign_dgsf10(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignAuxAlignDGSF10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignAuxAlignDGSF10 = FakeUNetFormerPreAlignAuxAlignDGSF10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_prealign_auxalign_dgsf10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["dgsf10"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignAuxAlignDGSF10)
            self.assertEqual(
                captured_kwargs,
                [
                    {
                        "num_classes": 6,
                        "sam_backbone": "vit_b",
                        "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                        "record_intermediate_stats": True,
                        "record_intermediate_modules": ["dgsf10"],
                    }
                ],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_passes_dga_intermediate_stats_flag(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA10 = FakeUNetFormerDGA10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_dga10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA10)
            self.assertTrue(captured_kwargs[0]["record_intermediate_stats"])
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_passes_record_intermediate_modules_when_configured(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA20DGSF10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA20DGSF10 = FakeUNetFormerDGA20DGSF10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_dga20_dgsf10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                    "record_intermediate_modules": ["dgsf10"],
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA20DGSF10)
            self.assertEqual(captured_kwargs[0]["record_intermediate_modules"], ["dgsf10"])
            self.assertTrue(captured_kwargs[0]["record_intermediate_stats"])
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_independent_dga_softplus(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA10Softplus:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA10Softplus = FakeUNetFormerDGA10Softplus
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_dga10_softplus",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                    "record_intermediate_stats": True,
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA10Softplus)
            self.assertTrue(captured_kwargs[0]["record_intermediate_stats"])
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_rejects_retired_dga_type_names(self) -> None:
        build_module = importlib.import_module("models.build")
        base_cfg = {
            "num_classes": 6,
            "sam_backbone": "vit_b",
            "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
        }

        for model_type in [
            "mfnet_unetformer_dga",
            "mfnet_unetformer_dga2",
            "mfnet_unetformer_dga3",
            "mfnet_unetformer_dga10_contrib_stats",
            "mfnet_unetformer_dga20_contrib_stats",
            "mfnet_unetformer_prealign_dga",
        ]:
            with self.subTest(model_type=model_type):
                with self.assertRaises(KeyError):
                    build_module.build_model({"type": model_type, **base_cfg})

    def test_build_model_dispatches_to_mfnet_dga30(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerDGA30:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerDGA30 = FakeUNetFormerDGA30
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_dga30",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerDGA30)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlign:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlign = FakeUNetFormerPreAlign
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_prealign",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlign)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign_mmadapter10(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignMMAdapter10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignMMAdapter10 = FakeUNetFormerPreAlignMMAdapter10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_prealign_mmadapter10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignMMAdapter10)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign_mmadapter20(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignMMAdapter20:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignMMAdapter20 = FakeUNetFormerPreAlignMMAdapter20
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_prealign_mmadapter20",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignMMAdapter20)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign_mmadapter21(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignMMAdapter21:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignMMAdapter21 = FakeUNetFormerPreAlignMMAdapter21
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_prealign_mmadapter21",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignMMAdapter21)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_auxalign(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerAuxAlign:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerAuxAlign = FakeUNetFormerAuxAlign
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_auxalign",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerAuxAlign)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign_auxalign(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignAuxAlign:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignAuxAlign = FakeUNetFormerPreAlignAuxAlign
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_prealign_auxalign",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignAuxAlign)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign_dga(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignDGA10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignDGA10 = FakeUNetFormerPreAlignDGA10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_prealign_dga10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignDGA10)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module

    def test_build_model_dispatches_to_mfnet_prealign_auxalign_dga(self) -> None:
        build_module = importlib.import_module("models.build")
        captured_kwargs: list[dict[str, object]] = []

        class FakeUNetFormerPreAlignAuxAlignDGA10:
            def __init__(self, **kwargs: object) -> None:
                captured_kwargs.append(kwargs)

        fake_mfnet_module = types.ModuleType("models.mfnet")
        fake_mfnet_module.UNetFormerPreAlignAuxAlignDGA10 = FakeUNetFormerPreAlignAuxAlignDGA10
        original_mfnet_module = sys.modules.get("models.mfnet")

        try:
            sys.modules["models.mfnet"] = fake_mfnet_module

            model = build_module.build_model(
                {
                    "type": "testnet_prealign_auxalign_dga10",
                    "num_classes": 6,
                    "sam_backbone": "vit_b",
                    "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth",
                }
            )

            self.assertIsInstance(model, FakeUNetFormerPreAlignAuxAlignDGA10)
            self.assertEqual(
                captured_kwargs,
                [{"num_classes": 6, "sam_backbone": "vit_b", "sam_checkpoint": "/tmp/sam_vit_b_01ec64.pth"}],
            )
        finally:
            if original_mfnet_module is None:
                del sys.modules["models.mfnet"]
            else:
                sys.modules["models.mfnet"] = original_mfnet_module
