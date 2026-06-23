from __future__ import annotations

import types
import unittest
from unittest.mock import patch

import torch

from multimodal_helpers import _fake_mfnet_optional_imports
from utils import IntermediateStatsRecorder


class MFNetIntermediateStatsConfigTest(unittest.TestCase):
    def test_attach_intermediate_stats_sets_recorder_and_prefix(self) -> None:
        from models.mfnet.intermediate_stats_config import attach_intermediate_stats

        owner = torch.nn.Module()
        owner.intermediate_stats = IntermediateStatsRecorder()
        module = torch.nn.Module()

        attach_intermediate_stats(owner, module, "dgsf10")

        self.assertIs(module.intermediate_stats, owner.intermediate_stats)
        self.assertEqual(module.intermediate_stats_prefix, "dgsf10")

    def test_attach_requested_intermediate_stats_mounts_requested_modules_only(self) -> None:
        from models.mfnet.intermediate_stats_config import attach_requested_intermediate_stats

        owner = torch.nn.Module()
        dga_block_0 = torch.nn.Module()
        dga_block_1 = torch.nn.Module()
        dgsf10 = torch.nn.Module()

        attach_requested_intermediate_stats(
            owner,
            ["unknown", "dga"],
            {
                "dga": [
                    (dga_block_0, "dga/block_0"),
                    (dga_block_1, "dga/block_1"),
                ],
                "dgsf10": [(dgsf10, "dgsf10")],
            },
        )

        self.assertIs(dga_block_0.intermediate_stats, owner.intermediate_stats)
        self.assertEqual(dga_block_0.intermediate_stats_prefix, "dga/block_0")
        self.assertIs(dga_block_1.intermediate_stats, owner.intermediate_stats)
        self.assertEqual(dga_block_1.intermediate_stats_prefix, "dga/block_1")
        self.assertFalse(hasattr(dgsf10, "intermediate_stats"))

        empty_owner = torch.nn.Module()
        unavailable_module = torch.nn.Module()
        attach_requested_intermediate_stats(
            empty_owner,
            ["unknown"],
            {"dga": [(unavailable_module, "dga/block_0")]},
        )

        self.assertFalse(hasattr(empty_owner, "intermediate_stats"))
        self.assertFalse(hasattr(unavailable_module, "intermediate_stats"))

    def test_unetformer_dgsf10_records_dgsf10_intermediate_stats_when_requested(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgsf10 import UNetFormerDGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dgsf10.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGSF10(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dgsf10"],
                )

        self.assertIsNotNone(model.intermediate_stats)
        self.assertIs(model.dgsf10.intermediate_stats, model.intermediate_stats)

    def test_unetformer_dgsf10_records_no_intermediate_modules_by_default(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgsf10 import UNetFormerDGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dgsf10.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGSF10(record_intermediate_stats=True)

        self.assertFalse(hasattr(model, "intermediate_stats"))
        self.assertFalse(hasattr(model.dgsf10, "intermediate_stats"))

    def test_unetformer_dgsf10_ignores_unavailable_intermediate_stats_modules(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgsf10 import UNetFormerDGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dgsf10.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGSF10(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dga"],
                )

        self.assertFalse(hasattr(model, "intermediate_stats"))
        self.assertFalse(hasattr(model.dgsf10, "intermediate_stats"))

    def test_unetformer_prealign_auxalign_dgsf10_records_dgsf10_stats_when_requested(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_prealign_auxalign_dgsf10 import (
                UNetFormerPreAlignAuxAlignDGSF10,
            )

            def fake_parent_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )
                self.align_index = 0

            with patch(
                "models.mfnet.UNetFormer_MMSAM_prealign_auxalign_dgsf10.UNetFormerPreAlignAuxAlign.__init__",
                new=fake_parent_init,
            ):
                model = UNetFormerPreAlignAuxAlignDGSF10(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dgsf10"],
                )

        self.assertIsNotNone(model.intermediate_stats)
        self.assertIs(model.dgsf10.intermediate_stats, model.intermediate_stats)
        self.assertEqual(model.dgsf10.intermediate_stats_prefix, "dgsf10")

    def test_unetformer_prealign_auxalign_dgsf10_records_no_stats_by_default(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_prealign_auxalign_dgsf10 import (
                UNetFormerPreAlignAuxAlignDGSF10,
            )

            def fake_parent_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )
                self.align_index = 0

            with patch(
                "models.mfnet.UNetFormer_MMSAM_prealign_auxalign_dgsf10.UNetFormerPreAlignAuxAlign.__init__",
                new=fake_parent_init,
            ):
                model = UNetFormerPreAlignAuxAlignDGSF10(record_intermediate_stats=True)

        self.assertFalse(hasattr(model, "intermediate_stats"))
        self.assertFalse(hasattr(model.dgsf10, "intermediate_stats"))

    def test_unetformer_dga20_dgsf10_records_no_intermediate_modules_by_default(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dga20_dgsf10 import UNetFormerDGA20DGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dga20.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGA20DGSF10(record_intermediate_stats=True)

        self.assertFalse(hasattr(model, "intermediate_stats"))
        for block in model.dga_blocks:
            self.assertFalse(hasattr(block, "intermediate_stats"))
        self.assertFalse(hasattr(model.dgsf10, "intermediate_stats"))

    def test_unetformer_dga20_dgsf10_initializes_parent_without_intermediate_stats(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dga20_dgsf10 import UNetFormerDGA20DGSF10

            captured_kwargs: list[dict[str, object]] = []

            def fake_dga20_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args
                captured_kwargs.append(kwargs)
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(embed_dim=8)
                self.dga_blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(4)])

            with patch("models.mfnet.UNetFormer_MMSAM_dga20_dgsf10.UNetFormerDGA20.__init__", new=fake_dga20_init):
                model = UNetFormerDGA20DGSF10(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dga", "dgsf10"],
                )

        self.assertEqual(captured_kwargs[0]["record_intermediate_stats"], False)
        self.assertEqual(captured_kwargs[0]["record_intermediate_modules"], ())
        for block in model.dga_blocks:
            self.assertIs(block.intermediate_stats, model.intermediate_stats)
        self.assertIs(model.dgsf10.intermediate_stats, model.intermediate_stats)

    def test_unetformer_dga20_dgsf10_can_record_only_dga_intermediate_stats(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dga20_dgsf10 import UNetFormerDGA20DGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dga20.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGA20DGSF10(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dga"],
                )

        for block in model.dga_blocks:
            self.assertIs(block.intermediate_stats, model.intermediate_stats)
        self.assertFalse(hasattr(model.dgsf10, "intermediate_stats"))

    def test_unetformer_dga20_dgsf10_can_record_only_dgsf10_intermediate_stats(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dga20_dgsf10 import UNetFormerDGA20DGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dga20.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGA20DGSF10(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["unknown", "dgsf10"],
                )

        for block in model.dga_blocks:
            self.assertFalse(hasattr(block, "intermediate_stats"))
        self.assertIs(model.dgsf10.intermediate_stats, model.intermediate_stats)

    def test_unetformer_dgfm01_records_dgfm01_intermediate_stats_when_requested(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgfm01 import UNetFormerDGFM01

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[
                        types.SimpleNamespace(window_size=0),
                        types.SimpleNamespace(window_size=14),
                        types.SimpleNamespace(window_size=0),
                        types.SimpleNamespace(window_size=0),
                        types.SimpleNamespace(window_size=14),
                        types.SimpleNamespace(window_size=14),
                    ],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dgfm01.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGFM01(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dgfm01"],
                )

        self.assertIsNotNone(model.intermediate_stats)
        for index, block in enumerate(model.dgfm01_blocks):
            self.assertIs(block.intermediate_stats, model.intermediate_stats)
            self.assertEqual(block.intermediate_stats_prefix, f"dgfm01/block_{index}")

    def test_unetformer_dgfm01_upernet_records_dgfm01_upernet_stats_when_requested(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgfm01_upernet import UNetFormerDGFM01UperNet

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[
                        types.SimpleNamespace(window_size=0),
                        types.SimpleNamespace(window_size=14),
                        types.SimpleNamespace(window_size=0),
                        types.SimpleNamespace(window_size=0),
                        types.SimpleNamespace(window_size=14),
                        types.SimpleNamespace(window_size=14),
                    ],
                )
                self.decoder = torch.nn.Identity()

            with patch(
                "models.mfnet.UNetFormer_MMSAM_dgfm01_upernet.UNetFormer.__init__",
                new=fake_unetformer_init,
            ):
                model = UNetFormerDGFM01UperNet(
                    sam_backbone="vit_b",
                    sam_checkpoint="/tmp/sam.pth",
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dgfm01_upernet"],
                )

        self.assertIsNotNone(model.intermediate_stats)
        for index, block in enumerate(model.dgfm01_upernet_blocks):
            self.assertIs(block.intermediate_stats, model.intermediate_stats)
            self.assertEqual(block.intermediate_stats_prefix, f"dgfm01_upernet/block_{index}")

    def test_unetformer_dga20_ignores_unavailable_intermediate_stats_modules(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dga20 import UNetFormerDGA20

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dga20.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGA20(
                    record_intermediate_stats=True,
                    record_intermediate_modules=["dgsf10"],
                )

        self.assertFalse(hasattr(model, "intermediate_stats"))
        for block in model.dga_blocks:
            self.assertFalse(hasattr(block, "intermediate_stats"))

    def test_record_intermediate_stats_false_disables_requested_modules(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dga20_dgsf10 import UNetFormerDGA20DGSF10

            def fake_unetformer_init(self: torch.nn.Module, *args: object, **kwargs: object) -> None:
                del args, kwargs
                torch.nn.Module.__init__(self)
                self.image_encoder = types.SimpleNamespace(
                    embed_dim=8,
                    blocks=[types.SimpleNamespace(window_size=0) for _ in range(4)],
                )

            with patch("models.mfnet.UNetFormer_MMSAM_dga20.UNetFormer.__init__", new=fake_unetformer_init):
                model = UNetFormerDGA20DGSF10(
                    record_intermediate_stats=False,
                    record_intermediate_modules=["dga", "dgsf10"],
                )

        self.assertFalse(hasattr(model, "intermediate_stats"))
        for block in model.dga_blocks:
            self.assertFalse(hasattr(block, "intermediate_stats"))
        self.assertFalse(hasattr(model.dgsf10, "intermediate_stats"))
