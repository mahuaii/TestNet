from __future__ import annotations

import sys
import types
import unittest

import torch

from multimodal_helpers import _fake_mfnet_optional_imports


class MFNetForwardContractTest(unittest.TestCase):
    def test_unetformer_prealign_expands_auxiliary_input_before_encoder(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_prealign import UNetFormerPreAlign
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        class FakeAuxPreAlign(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shape: tuple[int, ...] | None = None

            def forward(self, y: torch.Tensor) -> torch.Tensor:
                self.input_shape = tuple(y.shape)
                return y.repeat(1, 3, 1, 1)

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.aux_shape: tuple[int, ...] | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                self.aux_shape = tuple(y.shape)
                batch_size = x.shape[0]
                return torch.ones(batch_size, 256, 2, 2), torch.ones(batch_size, 256, 2, 2)

        class FakeFusion(torch.nn.Module):
            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
                return rgb + aux

        class FakeDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                del res1, res2, res3, res4
                self.output_size = (h, w)
                return torch.zeros(2, 6, h, w)

        model = UNetFormerPreAlign.__new__(UNetFormerPreAlign)
        torch.nn.Module.__init__(model)
        model.aux_prealign = FakeAuxPreAlign()
        model.image_encoder = FakeImageEncoder()
        model.fpn1x = torch.nn.Identity()
        model.fpn2x = torch.nn.Identity()
        model.fpn3x = torch.nn.Identity()
        model.fpn4x = torch.nn.Identity()
        model.fpn1y = torch.nn.Identity()
        model.fpn2y = torch.nn.Identity()
        model.fpn3y = torch.nn.Identity()
        model.fpn4y = torch.nn.Identity()
        model.fusion1 = FakeFusion()
        model.fusion2 = FakeFusion()
        model.fusion3 = FakeFusion()
        model.fusion4 = FakeFusion()
        model.decoder = FakeDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        self.assertEqual(model.aux_prealign.input_shape, (2, 1, 8, 8))
        self.assertEqual(model.image_encoder.aux_shape, (2, 3, 8, 8))
        self.assertEqual(model.decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_dga_repeats_auxiliary_input_and_applies_dga(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_dga10 import UNetFormerDGA10
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shapes: list[tuple[int, ...]] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.input_shapes.append(tuple(x.shape))
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.input_rgb: torch.Tensor | None = None
                self.input_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.input_rgb = x.detach().clone()
                self.input_aux = y.detach().clone()
                return x + self.rgb_offset, y + self.aux_offset

        class FakeNeck(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append("neck")
                self.inputs.append(x.detach().clone())
                return x

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=14, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=0, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = FakeNeck()

        class FakeFPN(torch.nn.Module):
            def __init__(self, name: str, offset: float) -> None:
                super().__init__()
                self.name = name
                self.offset = offset
                self.last_output: torch.Tensor | None = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                self.last_output = x + self.offset
                return self.last_output

        class SpyDGA(torch.nn.Module):
            def __init__(self, name: str, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                self.output_rgb = rgb.detach().clone() + self.rgb_offset
                self.output_aux = aux.detach().clone() + self.aux_offset
                return rgb + self.rgb_offset, aux + self.aux_offset

        class SpyFusion(torch.nn.Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                return rgb + aux

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                del res2, res3, res4
                events.append("decoder")
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerDGA10.__new__(UNetFormerDGA10)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.dga_indexes = [0, 2, 4, 5]
        model.dga_blocks = torch.nn.ModuleList(
            [
                SpyDGA("dga10_0", rgb_offset=100.0, aux_offset=1000.0),
                SpyDGA("dga10_1", rgb_offset=200.0, aux_offset=2000.0),
                SpyDGA("dga10_2", rgb_offset=300.0, aux_offset=3000.0),
                SpyDGA("dga10_3", rgb_offset=400.0, aux_offset=4000.0),
            ]
        )
        model.fpn1x = FakeFPN("fpn1x", 1.0)
        model.fpn2x = FakeFPN("fpn2x", 2.0)
        model.fpn3x = FakeFPN("fpn3x", 3.0)
        model.fpn4x = FakeFPN("fpn4x", 4.0)
        model.fpn1y = FakeFPN("fpn1y", 10.0)
        model.fpn2y = FakeFPN("fpn2y", 20.0)
        model.fpn3y = FakeFPN("fpn3y", 30.0)
        model.fpn4y = FakeFPN("fpn4y", 40.0)
        model.fusion1 = SpyFusion("fusion1")
        model.fusion2 = SpyFusion("fusion2")
        model.fusion3 = SpyFusion("fusion3")
        model.fusion4 = SpyFusion("fusion4")
        model.decoder = SpyDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        self.assertFalse(hasattr(model, "aux_prealign"))
        self.assertEqual(model.image_encoder.patch_embed.input_shapes[1], (2, 3, 8, 8))
        self.assertEqual([len(dga_block.calls) for dga_block in model.dga_blocks], [1, 1, 1, 1])
        for dga_block in model.dga_blocks:
            self.assertEqual(dga_block.calls[0][0].shape, (2, 4, 2, 2))
            self.assertEqual(dga_block.calls[0][1].shape, (2, 4, 2, 2))

        expected_order = [
            "block0",
            "dga10_0",
            "block1",
            "block2",
            "dga10_1",
            "block3",
            "block4",
            "dga10_2",
            "block5",
            "dga10_3",
        ]
        self.assertEqual(events[: len(expected_order)], expected_order)
        self.assertLess(events.index("dga10_3"), events.index("neck"))
        self.assertEqual(model.decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_dga2_applies_dga2_after_global_blocks_with_bchw_boundary(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_dga20 import UNetFormerDGA20
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shapes: list[tuple[int, ...]] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.input_shapes.append(tuple(x.shape))
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.input_rgb: torch.Tensor | None = None
                self.input_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.input_rgb = x.detach().clone()
                self.input_aux = y.detach().clone()
                return x + self.rgb_offset, y + self.aux_offset

        class FakeNeck(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append("neck")
                self.inputs.append(x.detach().clone())
                return x

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=14, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=0, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = FakeNeck()

        class SpyDGA2(torch.nn.Module):
            def __init__(self, name: str, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None
                self.alpha = torch.nn.Parameter(torch.tensor([0.1]))
                self.beta = torch.nn.Parameter(torch.tensor([0.1]))

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                self.output_rgb = rgb.detach().clone() + self.rgb_offset
                self.output_aux = aux.detach().clone() + self.aux_offset
                return rgb + self.rgb_offset, aux + self.aux_offset

        class FakeFPN(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        class FakeFusion(torch.nn.Module):
            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return x + y

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                del res2, res3, res4
                events.append("decoder")
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerDGA20.__new__(UNetFormerDGA20)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.dga_indexes = [0, 2, 4, 5]
        model.dga_blocks = torch.nn.ModuleList(
            [
                SpyDGA2("dga2_0", rgb_offset=100.0, aux_offset=1000.0),
                SpyDGA2("dga20", rgb_offset=200.0, aux_offset=2000.0),
                SpyDGA2("dga2_2", rgb_offset=300.0, aux_offset=3000.0),
                SpyDGA2("dga2_3", rgb_offset=400.0, aux_offset=4000.0),
            ]
        )
        model.fpn1x = FakeFPN()
        model.fpn2x = FakeFPN()
        model.fpn3x = FakeFPN()
        model.fpn4x = FakeFPN()
        model.fpn1y = FakeFPN()
        model.fpn2y = FakeFPN()
        model.fpn3y = FakeFPN()
        model.fpn4y = FakeFPN()
        model.fusion1 = FakeFusion()
        model.fusion2 = FakeFusion()
        model.fusion3 = FakeFusion()
        model.fusion4 = FakeFusion()
        model.decoder = SpyDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        self.assertEqual(model.image_encoder.patch_embed.input_shapes[1], (2, 3, 8, 8))
        self.assertEqual([len(dga_block.calls) for dga_block in model.dga_blocks], [1, 1, 1, 1])
        for dga_block in model.dga_blocks:
            self.assertEqual(dga_block.calls[0][0].shape, (2, 4, 2, 2))
            self.assertEqual(dga_block.calls[0][1].shape, (2, 4, 2, 2))

        expected_order = [
            "block0",
            "dga2_0",
            "block1",
            "block2",
            "dga20",
            "block3",
            "block4",
            "dga2_2",
            "block5",
            "dga2_3",
        ]
        self.assertEqual(events[: len(expected_order)], expected_order)
        self.assertLess(events.index("dga2_3"), events.index("neck"))

        block1 = model.image_encoder.blocks[1]
        block3 = model.image_encoder.blocks[3]
        block5 = model.image_encoder.blocks[5]
        self.assertTrue(torch.equal(block1.input_rgb, model.dga_blocks[0].output_rgb.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block1.input_aux, model.dga_blocks[0].output_aux.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block3.input_rgb, model.dga_blocks[1].output_rgb.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block3.input_aux, model.dga_blocks[1].output_aux.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block5.input_rgb, model.dga_blocks[2].output_rgb.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block5.input_aux, model.dga_blocks[2].output_aux.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(model.image_encoder.neck.inputs[0], model.dga_blocks[3].output_rgb))
        self.assertTrue(torch.equal(model.image_encoder.neck.inputs[1], model.dga_blocks[3].output_aux))
        self.assertEqual(model.decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_dga20_dgsf10_uses_dga_features_and_encoder_final_top(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_dga20_dgsf10 import UNetFormerDGA20DGSF10
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.output_rgb = x.detach().clone() + self.rgb_offset
                self.output_aux = y.detach().clone() + self.aux_offset
                return x + self.rgb_offset, y + self.aux_offset

        class ForbiddenNeck(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                del x
                raise AssertionError("DGSF10 top features must be captured before encoder.neck")

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=14, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = ForbiddenNeck()

        class SpyDGA(torch.nn.Module):
            def __init__(self, name: str, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.output_rgb = rgb.detach().clone() + self.rgb_offset
                self.output_aux = aux.detach().clone() + self.aux_offset
                return rgb + self.rgb_offset, aux + self.aux_offset

        class SpyDGSF10(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rgb_feats: tuple[torch.Tensor, ...] | None = None
                self.aux_feats: tuple[torch.Tensor, ...] | None = None
                self.outputs = tuple(torch.full((2, 4, 2, 2), float(index)) for index in range(1, 5))

            def forward(
                self,
                rgb_feats: tuple[torch.Tensor, ...],
                aux_feats: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                events.append("dgsf10")
                self.rgb_feats = tuple(feature.detach().clone() for feature in rgb_feats)
                self.aux_feats = tuple(feature.detach().clone() for feature in aux_feats)
                return self.outputs

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1, res2, res3, res4)
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerDGA20DGSF10.__new__(UNetFormerDGA20DGSF10)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.dga_indexes = [0, 2, 3, 4]
        model.dga_blocks = torch.nn.ModuleList(
            [
                SpyDGA("dga0", rgb_offset=100.0, aux_offset=1000.0),
                SpyDGA("dga1", rgb_offset=200.0, aux_offset=2000.0),
                SpyDGA("dga2", rgb_offset=300.0, aux_offset=3000.0),
                SpyDGA("dga3", rgb_offset=400.0, aux_offset=4000.0),
            ]
        )
        model.dgsf10 = SpyDGSF10()
        model.decoder = SpyDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        dgsf10 = model.dgsf10
        decoder = model.decoder
        assert isinstance(dgsf10, SpyDGSF10)
        assert isinstance(decoder, SpyDecoder)
        self.assertEqual(
            events,
            [
                "block0",
                "dga0",
                "block1",
                "block2",
                "dga1",
                "block3",
                "dga2",
                "block4",
                "dga3",
                "block5",
                "dgsf10",
                "decoder",
            ],
        )
        self.assertIsNotNone(dgsf10.rgb_feats)
        self.assertIsNotNone(dgsf10.aux_feats)
        self.assertEqual(len(dgsf10.rgb_feats), 5)
        self.assertEqual(len(dgsf10.aux_feats), 5)
        for index, dga_block in enumerate(model.dga_blocks):
            assert isinstance(dga_block, SpyDGA)
            self.assertTrue(torch.equal(dgsf10.rgb_feats[index], dga_block.output_rgb))
            self.assertTrue(torch.equal(dgsf10.aux_feats[index], dga_block.output_aux))
        final_block = model.image_encoder.blocks[-1]
        assert isinstance(final_block, FakeBlock)
        self.assertIsNotNone(final_block.output_rgb)
        self.assertIsNotNone(final_block.output_aux)
        self.assertTrue(torch.equal(dgsf10.rgb_feats[4], final_block.output_rgb.permute(0, 3, 1, 2)))
        self.assertTrue(torch.equal(dgsf10.aux_feats[4], final_block.output_aux.permute(0, 3, 1, 2)))
        self.assertIsNotNone(decoder.inputs)
        for actual, expected in zip(decoder.inputs, dgsf10.outputs):
            self.assertIs(actual, expected)
        self.assertEqual(decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_dgsf10_uses_encoder_features_and_encoder_final_top(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgsf10 import UNetFormerDGSF10

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.output_rgb = x.detach().clone() + self.rgb_offset
                self.output_aux = y.detach().clone() + self.aux_offset
                return x + self.rgb_offset, y + self.aux_offset

        class ForbiddenNeck(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                del x
                raise AssertionError("DGSF10 top features must be captured before encoder.neck")

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=14, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = ForbiddenNeck()

        class SpyDGSF10(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rgb_feats: tuple[torch.Tensor, ...] | None = None
                self.aux_feats: tuple[torch.Tensor, ...] | None = None
                self.outputs = tuple(torch.full((2, 4, 2, 2), float(index)) for index in range(1, 5))

            def forward(
                self,
                rgb_feats: tuple[torch.Tensor, ...],
                aux_feats: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                events.append("dgsf10")
                self.rgb_feats = tuple(feature.detach().clone() for feature in rgb_feats)
                self.aux_feats = tuple(feature.detach().clone() for feature in aux_feats)
                return self.outputs

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1, res2, res3, res4)
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerDGSF10.__new__(UNetFormerDGSF10)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.dgsf10_indexes = [0, 2, 3, 4]
        model.dgsf10 = SpyDGSF10()
        model.decoder = SpyDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        dgsf10 = model.dgsf10
        decoder = model.decoder
        assert isinstance(dgsf10, SpyDGSF10)
        assert isinstance(decoder, SpyDecoder)
        self.assertEqual(events, ["block0", "block1", "block2", "block3", "block4", "block5", "dgsf10", "decoder"])
        self.assertIsNotNone(dgsf10.rgb_feats)
        self.assertIsNotNone(dgsf10.aux_feats)
        self.assertEqual(len(dgsf10.rgb_feats), 5)
        self.assertEqual(len(dgsf10.aux_feats), 5)
        for feature_index, block_index in enumerate(model.dgsf10_indexes):
            block = model.image_encoder.blocks[block_index]
            assert isinstance(block, FakeBlock)
            self.assertIsNotNone(block.output_rgb)
            self.assertIsNotNone(block.output_aux)
            self.assertTrue(torch.equal(dgsf10.rgb_feats[feature_index], block.output_rgb.permute(0, 3, 1, 2)))
            self.assertTrue(torch.equal(dgsf10.aux_feats[feature_index], block.output_aux.permute(0, 3, 1, 2)))
        final_block = model.image_encoder.blocks[-1]
        assert isinstance(final_block, FakeBlock)
        self.assertIsNotNone(final_block.output_rgb)
        self.assertIsNotNone(final_block.output_aux)
        self.assertTrue(torch.equal(dgsf10.rgb_feats[4], final_block.output_rgb.permute(0, 3, 1, 2)))
        self.assertTrue(torch.equal(dgsf10.aux_feats[4], final_block.output_aux.permute(0, 3, 1, 2)))
        self.assertIsNotNone(decoder.inputs)
        for actual, expected in zip(decoder.inputs, dgsf10.outputs):
            self.assertIs(actual, expected)
        self.assertEqual(decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_dgfm_uses_first_three_global_blocks_and_deepest_outputs(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgfm import UNetFormerDGFM

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shapes: list[tuple[int, ...]] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.input_shapes.append(tuple(x.shape))
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.output_rgb = x.detach().clone() + self.rgb_offset
                self.output_aux = y.detach().clone() + self.aux_offset
                return x + self.rgb_offset, y + self.aux_offset

        class ForbiddenNeck(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                del x
                raise AssertionError("DGFM decoder features must be captured before encoder.neck")

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=14, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = ForbiddenNeck()

        class SpyDGFM(torch.nn.Module):
            def __init__(self, name: str, base_value: float) -> None:
                super().__init__()
                self.name = name
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
                self.outputs = tuple(
                    torch.full((2, 4, 2, 2), base_value + float(index)) for index in range(4)
                )

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, ...]:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                return self.outputs

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1, res2, res3, res4)
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerDGFM.__new__(UNetFormerDGFM)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.dgfm_indexes = [0, 2, 3, 5]
        model.dgfm_blocks = torch.nn.ModuleList(
            [
                SpyDGFM("dgfm0", base_value=10.0),
                SpyDGFM("dgfm1", base_value=20.0),
                SpyDGFM("dgfm2", base_value=30.0),
                SpyDGFM("dgfm3", base_value=40.0),
            ]
        )
        model.decoder = SpyDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        decoder = model.decoder
        assert isinstance(decoder, SpyDecoder)
        self.assertFalse(hasattr(model, "fusion1"))
        self.assertEqual(model.image_encoder.patch_embed.input_shapes[0], (2, 3, 8, 8))
        self.assertEqual(model.image_encoder.patch_embed.input_shapes[1], (2, 3, 8, 8))
        self.assertEqual(
            events,
            [
                "block0",
                "dgfm0",
                "block1",
                "block2",
                "dgfm1",
                "block3",
                "dgfm2",
                "block4",
                "block5",
                "dgfm3",
                "decoder",
            ],
        )
        self.assertIsNotNone(decoder.inputs)
        for tap_index, dgfm in enumerate(model.dgfm_blocks):
            assert isinstance(dgfm, SpyDGFM)
            self.assertEqual(len(dgfm.calls), 1)
            self.assertIs(decoder.inputs[tap_index], dgfm.outputs[tap_index])
        self.assertEqual(decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_sgcf_uses_dgfm_style_taps_and_raw_dsm(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_sgcf import UNetFormerSGCF

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shapes: list[tuple[int, ...]] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.input_shapes.append(tuple(x.shape))
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                return x + self.rgb_offset, y + self.aux_offset

        class ForbiddenNeck(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                del x
                raise AssertionError("SGCF decoder features must be captured before encoder.neck")

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=14, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = ForbiddenNeck()

        class SpySGCF(torch.nn.Module):
            def __init__(self, name: str, base_value: float) -> None:
                super().__init__()
                self.name = name
                self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
                self.outputs = tuple(torch.full((2, 4, 2, 2), base_value + float(index)) for index in range(4))

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor, dsm: torch.Tensor) -> tuple[torch.Tensor, ...]:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone(), dsm.detach().clone()))
                return self.outputs

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1, res2, res3, res4)
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerSGCF.__new__(UNetFormerSGCF)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.sgcf_indexes = [0, 2, 3, 5]
        model.sgcf_blocks = torch.nn.ModuleList(
            [
                SpySGCF("sgcf0", base_value=10.0),
                SpySGCF("sgcf1", base_value=20.0),
                SpySGCF("sgcf2", base_value=30.0),
                SpySGCF("sgcf3", base_value=40.0),
            ]
        )
        model.decoder = SpyDecoder()
        raw_dsm = torch.ones(2, 8, 8)

        output = model(torch.zeros(2, 3, 8, 8), raw_dsm)

        decoder = model.decoder
        assert isinstance(decoder, SpyDecoder)
        self.assertFalse(hasattr(model, "fusion1"))
        self.assertEqual(model.image_encoder.patch_embed.input_shapes[0], (2, 3, 8, 8))
        self.assertEqual(model.image_encoder.patch_embed.input_shapes[1], (2, 3, 8, 8))
        self.assertEqual(
            events,
            [
                "block0",
                "sgcf0",
                "block1",
                "block2",
                "sgcf1",
                "block3",
                "sgcf2",
                "block4",
                "block5",
                "sgcf3",
                "decoder",
            ],
        )
        self.assertIsNotNone(decoder.inputs)
        for tap_index, sgcf in enumerate(model.sgcf_blocks):
            assert isinstance(sgcf, SpySGCF)
            self.assertEqual(len(sgcf.calls), 1)
            self.assertIs(decoder.inputs[tap_index], sgcf.outputs[tap_index])
            self.assertTrue(torch.equal(sgcf.calls[0][2], raw_dsm.unsqueeze(1)))
        self.assertEqual(decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_spmf10_uses_sam_taps_structure_branch_and_spmf10(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_spmf10 import UNetFormerSPMF10

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shapes: list[tuple[int, ...]] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.input_shapes.append(tuple(x.shape))
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, dsm_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.dsm_offset = dsm_offset
                self.output_dsm: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.output_dsm = y.detach().clone() + self.dsm_offset
                return x + self.rgb_offset, y + self.dsm_offset

        class FakeNeck(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append("neck")
                self.calls.append(x.detach().clone())
                return x + 100.0

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, dsm_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, dsm_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, dsm_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, dsm_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, dsm_offset=50.0),
                        FakeBlock("block5", window_size=14, rgb_offset=6.0, dsm_offset=60.0),
                    ]
                )
                self.neck = FakeNeck()

        class SpyFPN(torch.nn.Module):
            def __init__(self, name: str, output: torch.Tensor) -> None:
                super().__init__()
                self.name = name
                self.output = output
                self.calls: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                self.calls.append(x.detach().clone())
                return self.output

        class SpyStructureBranch(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dsm: torch.Tensor | None = None
                self.taps: tuple[torch.Tensor, ...] | None = None
                self.outputs = tuple(torch.full((2, 4, 2, 2), 200.0 + float(index)) for index in range(4))

            def forward(
                self,
                dsm: torch.Tensor,
                dsm_taps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                events.append("structure")
                self.dsm = dsm.detach().clone()
                self.taps = tuple(tap.detach().clone() for tap in dsm_taps)
                return self.outputs

        class SpySPMF10(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rgb_feats: tuple[torch.Tensor, ...] | None = None
                self.dsm_feats: tuple[torch.Tensor, ...] | None = None
                self.structure_feats: tuple[torch.Tensor, ...] | None = None
                self.outputs = tuple(torch.full((2, 4, 2, 2), 300.0 + float(index)) for index in range(4))

            def forward(
                self,
                rgb_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                dsm_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                structure_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                events.append("spmf10")
                self.rgb_feats = rgb_feats
                self.dsm_feats = dsm_feats
                self.structure_feats = structure_feats
                return self.outputs

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1, res2, res3, res4)
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerSPMF10.__new__(UNetFormerSPMF10)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.spmf10_indexes = [0, 2, 3, 4]
        fpn_outputs = tuple(torch.full((2, 4, 2, 2), 100.0 + float(index)) for index in range(8))
        model.fpn1x = SpyFPN("fpn1x", fpn_outputs[0])
        model.fpn2x = SpyFPN("fpn2x", fpn_outputs[1])
        model.fpn3x = SpyFPN("fpn3x", fpn_outputs[2])
        model.fpn4x = SpyFPN("fpn4x", fpn_outputs[3])
        model.fpn1y = SpyFPN("fpn1y", fpn_outputs[4])
        model.fpn2y = SpyFPN("fpn2y", fpn_outputs[5])
        model.fpn3y = SpyFPN("fpn3y", fpn_outputs[6])
        model.fpn4y = SpyFPN("fpn4y", fpn_outputs[7])
        model.structure_branch10 = SpyStructureBranch()
        model.spmf10 = SpySPMF10()
        model.decoder = SpyDecoder()
        raw_dsm = torch.ones(2, 8, 8)

        output = model(torch.zeros(2, 3, 8, 8), raw_dsm)

        structure_branch = model.structure_branch10
        spmf10 = model.spmf10
        decoder = model.decoder
        assert isinstance(structure_branch, SpyStructureBranch)
        assert isinstance(spmf10, SpySPMF10)
        assert isinstance(decoder, SpyDecoder)
        self.assertEqual(model.image_encoder.patch_embed.input_shapes[0], (2, 3, 8, 8))
        self.assertEqual(model.image_encoder.patch_embed.input_shapes[1], (2, 3, 8, 8))
        self.assertEqual(
            events,
            [
                "block0",
                "block1",
                "block2",
                "block3",
                "block4",
                "block5",
                "neck",
                "neck",
                "fpn1x",
                "fpn2x",
                "fpn3x",
                "fpn4x",
                "fpn1y",
                "fpn2y",
                "fpn3y",
                "fpn4y",
                "structure",
                "spmf10",
                "decoder",
            ],
        )
        self.assertIsNotNone(structure_branch.dsm)
        self.assertTrue(torch.equal(structure_branch.dsm, raw_dsm.unsqueeze(1)))
        self.assertIsNotNone(structure_branch.taps)
        for tap_index, block_index in enumerate(model.spmf10_indexes):
            block = model.image_encoder.blocks[block_index]
            assert isinstance(block, FakeBlock)
            self.assertIsNotNone(block.output_dsm)
            self.assertTrue(torch.equal(structure_branch.taps[tap_index], block.output_dsm.permute(0, 3, 1, 2)))
        self.assertIsNotNone(spmf10.rgb_feats)
        self.assertIsNotNone(spmf10.dsm_feats)
        self.assertIsNotNone(spmf10.structure_feats)
        for actual, expected in zip(spmf10.rgb_feats, fpn_outputs[:4]):
            self.assertIs(actual, expected)
        for actual, expected in zip(spmf10.dsm_feats, fpn_outputs[4:]):
            self.assertIs(actual, expected)
        for actual, expected in zip(spmf10.structure_feats, structure_branch.outputs):
            self.assertIs(actual, expected)
        self.assertIsNotNone(decoder.inputs)
        for actual, expected in zip(decoder.inputs, spmf10.outputs):
            self.assertIs(actual, expected)
        self.assertEqual(decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_prealign_spmf20_uses_prealigned_aux_for_sam_and_raw_dsm_for_structure(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_prealign_spmf20 import UNetFormerPreAlignSPMF20

        events: list[str] = []

        class FakeAuxPreAlign(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input: torch.Tensor | None = None

            def forward(self, y: torch.Tensor) -> torch.Tensor:
                events.append("prealign")
                self.input = y.detach().clone()
                return torch.cat([y + 1.0, y + 2.0, y + 3.0], dim=1)

        class FakePatchEmbed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.inputs.append(x.detach().clone())
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, dsm_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.dsm_offset = dsm_offset
                self.output_dsm: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.output_dsm = y.detach().clone() + self.dsm_offset
                return x + self.rgb_offset, y + self.dsm_offset

        class FakeNeck(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append("neck")
                return x + 100.0

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, dsm_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, dsm_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, dsm_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, dsm_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, dsm_offset=50.0),
                        FakeBlock("block5", window_size=14, rgb_offset=6.0, dsm_offset=60.0),
                    ]
                )
                self.neck = FakeNeck()

        class SpyFPN(torch.nn.Module):
            def __init__(self, name: str, output: torch.Tensor) -> None:
                super().__init__()
                self.name = name
                self.output = output

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                del x
                events.append(self.name)
                return self.output

        class SpyStructureBranch(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dsm: torch.Tensor | None = None
                self.taps: tuple[torch.Tensor, ...] | None = None
                self.outputs = tuple(torch.full((2, 4, 2, 2), 200.0 + float(index)) for index in range(4))

            def forward(
                self,
                dsm: torch.Tensor,
                dsm_taps: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                events.append("structure")
                self.dsm = dsm.detach().clone()
                self.taps = tuple(tap.detach().clone() for tap in dsm_taps)
                return self.outputs

        class SpySPMF20(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rgb_feats: tuple[torch.Tensor, ...] | None = None
                self.dsm_feats: tuple[torch.Tensor, ...] | None = None
                self.structure_feats: tuple[torch.Tensor, ...] | None = None
                self.outputs = tuple(torch.full((2, 4, 2, 2), 300.0 + float(index)) for index in range(4))

            def forward(
                self,
                rgb_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                dsm_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
                structure_feats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                events.append("spmf20")
                self.rgb_feats = rgb_feats
                self.dsm_feats = dsm_feats
                self.structure_feats = structure_feats
                return self.outputs

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1, res2, res3, res4)
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerPreAlignSPMF20.__new__(UNetFormerPreAlignSPMF20)
        torch.nn.Module.__init__(model)
        model.aux_prealign = FakeAuxPreAlign()
        model.image_encoder = FakeImageEncoder()
        model.spmf20_indexes = [0, 2, 3, 4]
        fpn_outputs = tuple(torch.full((2, 4, 2, 2), 100.0 + float(index)) for index in range(8))
        model.fpn1x = SpyFPN("fpn1x", fpn_outputs[0])
        model.fpn2x = SpyFPN("fpn2x", fpn_outputs[1])
        model.fpn3x = SpyFPN("fpn3x", fpn_outputs[2])
        model.fpn4x = SpyFPN("fpn4x", fpn_outputs[3])
        model.fpn1y = SpyFPN("fpn1y", fpn_outputs[4])
        model.fpn2y = SpyFPN("fpn2y", fpn_outputs[5])
        model.fpn3y = SpyFPN("fpn3y", fpn_outputs[6])
        model.fpn4y = SpyFPN("fpn4y", fpn_outputs[7])
        model.structure_branch10 = SpyStructureBranch()
        model.spmf20 = SpySPMF20()
        model.decoder = SpyDecoder()
        raw_dsm = torch.arange(128, dtype=torch.float32).view(2, 8, 8)

        output = model(torch.zeros(2, 3, 8, 8), raw_dsm)

        aux_prealign = model.aux_prealign
        patch_embed = model.image_encoder.patch_embed
        structure_branch = model.structure_branch10
        spmf20 = model.spmf20
        decoder = model.decoder
        assert isinstance(aux_prealign, FakeAuxPreAlign)
        assert isinstance(patch_embed, FakePatchEmbed)
        assert isinstance(structure_branch, SpyStructureBranch)
        assert isinstance(spmf20, SpySPMF20)
        assert isinstance(decoder, SpyDecoder)
        self.assertIsNotNone(aux_prealign.input)
        self.assertTrue(torch.equal(aux_prealign.input, raw_dsm.unsqueeze(1)))
        self.assertEqual(tuple(patch_embed.inputs[0].shape), (2, 3, 8, 8))
        self.assertTrue(torch.equal(patch_embed.inputs[1], torch.cat(
            [raw_dsm.unsqueeze(1) + 1.0, raw_dsm.unsqueeze(1) + 2.0, raw_dsm.unsqueeze(1) + 3.0],
            dim=1,
        )))
        self.assertEqual(
            events,
            [
                "prealign",
                "block0",
                "block1",
                "block2",
                "block3",
                "block4",
                "block5",
                "neck",
                "neck",
                "fpn1x",
                "fpn2x",
                "fpn3x",
                "fpn4x",
                "fpn1y",
                "fpn2y",
                "fpn3y",
                "fpn4y",
                "structure",
                "spmf20",
                "decoder",
            ],
        )
        self.assertIsNotNone(structure_branch.dsm)
        self.assertTrue(torch.equal(structure_branch.dsm, raw_dsm.unsqueeze(1)))
        self.assertIsNotNone(structure_branch.taps)
        for tap_index, block_index in enumerate(model.spmf20_indexes):
            block = model.image_encoder.blocks[block_index]
            assert isinstance(block, FakeBlock)
            self.assertIsNotNone(block.output_dsm)
            self.assertTrue(torch.equal(structure_branch.taps[tap_index], block.output_dsm.permute(0, 3, 1, 2)))
        self.assertIsNotNone(spmf20.rgb_feats)
        self.assertIsNotNone(spmf20.dsm_feats)
        self.assertIsNotNone(spmf20.structure_feats)
        for actual, expected in zip(spmf20.rgb_feats, fpn_outputs[:4]):
            self.assertIs(actual, expected)
        for actual, expected in zip(spmf20.dsm_feats, fpn_outputs[4:]):
            self.assertIs(actual, expected)
        for actual, expected in zip(spmf20.structure_feats, structure_branch.outputs):
            self.assertIs(actual, expected)
        self.assertIsNotNone(decoder.inputs)
        for actual, expected in zip(decoder.inputs, spmf20.outputs):
            self.assertIs(actual, expected)
        self.assertEqual(decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_dgfm01_applies_external_norm_projection_scale_and_decoder(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgfm01 import UNetFormerDGFM01

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shapes: list[tuple[int, ...]] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.input_shapes.append(tuple(x.shape))
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                return x + self.rgb_offset, y + self.aux_offset

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=14, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )

        class SpyNorm(torch.nn.Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name
                self.input_shapes: list[tuple[int, ...]] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                self.input_shapes.append(tuple(x.shape))
                return x

        class SpyDGFM01(torch.nn.Module):
            def __init__(self, name: str, base_value: float) -> None:
                super().__init__()
                self.name = name
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
                self.output = torch.full((2, 4, 2, 2), base_value)

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                return self.output

        class SpyProj(torch.nn.Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name
                self.input_shapes: list[tuple[int, ...]] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                self.input_shapes.append(tuple(x.shape))
                return x

        class SpyScaleAdapter(torch.nn.Module):
            def __init__(self, name: str, base_value: float) -> None:
                super().__init__()
                self.name = name
                self.calls: list[torch.Tensor] = []
                self.outputs = tuple(
                    torch.full((2, 4, 2, 2), base_value + float(index)) for index in range(4)
                )

            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
                events.append(self.name)
                self.calls.append(x.detach().clone())
                return self.outputs

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1, res2, res3, res4)
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerDGFM01.__new__(UNetFormerDGFM01)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.dgfm01_indexes = [0, 2, 3, 5]
        model.dgfm01_input_norms = torch.nn.ModuleList(
            [SpyNorm(f"input_norm{index}") for index in range(4)]
        )
        model.dgfm01_blocks = torch.nn.ModuleList(
            [SpyDGFM01(f"dgfm01_{index}", base_value=10.0 * (index + 1)) for index in range(4)]
        )
        model.dgfm01_output_norms = torch.nn.ModuleList(
            [SpyNorm(f"output_norm{index}") for index in range(4)]
        )
        model.dgfm01_output_projs = torch.nn.ModuleList(
            [SpyProj(f"output_proj{index}") for index in range(4)]
        )
        model.dgfm01_scale_adapters = torch.nn.ModuleList(
            [SpyScaleAdapter(f"scale_adapter{index}", base_value=100.0 * (index + 1)) for index in range(4)]
        )
        model.decoder = SpyDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        decoder = model.decoder
        assert isinstance(decoder, SpyDecoder)
        self.assertFalse(hasattr(model, "fusion1"))
        self.assertEqual(model.image_encoder.patch_embed.input_shapes[0], (2, 3, 8, 8))
        self.assertEqual(model.image_encoder.patch_embed.input_shapes[1], (2, 3, 8, 8))
        self.assertEqual(events[-1], "decoder")
        self.assertEqual(
            events[:9],
            [
                "block0",
                "input_norm0",
                "input_norm0",
                "dgfm01_0",
                "output_norm0",
                "output_proj0",
                "scale_adapter0",
                "block1",
                "block2",
            ],
        )
        self.assertIn("scale_adapter3", events)
        self.assertIsNotNone(decoder.inputs)
        for tap_index, scale_adapter in enumerate(model.dgfm01_scale_adapters):
            assert isinstance(scale_adapter, SpyScaleAdapter)
            self.assertEqual(len(scale_adapter.calls), 1)
            self.assertIs(decoder.inputs[tap_index], scale_adapter.outputs[tap_index])
        self.assertEqual(decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_dgfm01_upernet_applies_external_norm_projection_and_list_decoder(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgfm01_upernet import UNetFormerDGFM01UperNet

        events: list[str] = []

        class FakePatchEmbed(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shapes: list[tuple[int, ...]] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                self.input_shapes.append(tuple(x.shape))
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                return x + self.rgb_offset, y + self.aux_offset

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=14, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )

        class SpyNorm(torch.nn.Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                return x

        class SpyDGFM01(torch.nn.Module):
            def __init__(self, name: str, base_value: float) -> None:
                super().__init__()
                self.name = name
                self.output = torch.full((2, 4, 2, 2), base_value)

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
                del rgb, aux
                events.append(self.name)
                return self.output

        class SpyProj(torch.nn.Module):
            def __init__(self, name: str, base_value: float) -> None:
                super().__init__()
                self.name = name
                self.output = torch.full((2, 4, 2, 2), base_value)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                del x
                events.append(self.name)
                return self.output

        class SpyUperDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features: tuple[torch.Tensor, ...] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(self, features: tuple[torch.Tensor, ...], h: int, w: int) -> torch.Tensor:
                events.append("uper_decoder")
                self.features = tuple(features)
                self.output_size = (h, w)
                return torch.zeros(features[0].shape[0], 6, h, w)

        model = UNetFormerDGFM01UperNet.__new__(UNetFormerDGFM01UperNet)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()
        model.dgfm01_upernet_indexes = [0, 2, 3, 5]
        model.dgfm01_upernet_input_norms = torch.nn.ModuleList(
            [SpyNorm(f"input_norm{index}") for index in range(4)]
        )
        model.dgfm01_upernet_blocks = torch.nn.ModuleList(
            [SpyDGFM01(f"dgfm01_upernet_{index}", base_value=10.0 * (index + 1)) for index in range(4)]
        )
        model.dgfm01_upernet_output_norms = torch.nn.ModuleList(
            [SpyNorm(f"output_norm{index}") for index in range(4)]
        )
        model.dgfm01_upernet_output_projs = torch.nn.ModuleList(
            [SpyProj(f"output_proj{index}", base_value=100.0 * (index + 1)) for index in range(4)]
        )
        model.decoder = SpyUperDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        decoder = model.decoder
        assert isinstance(decoder, SpyUperDecoder)
        self.assertFalse(hasattr(model, "fusion1"))
        self.assertFalse(hasattr(model, "dgfm01_upernet_scale_adapters"))
        self.assertEqual(events[-1], "uper_decoder")
        self.assertEqual(
            events[:8],
            [
                "block0",
                "input_norm0",
                "input_norm0",
                "dgfm01_upernet_0",
                "output_norm0",
                "output_proj0",
                "block1",
                "block2",
            ],
        )
        self.assertIsNotNone(decoder.features)
        for tap_index, proj in enumerate(model.dgfm01_upernet_output_projs):
            assert isinstance(proj, SpyProj)
            self.assertIs(decoder.features[tap_index], proj.output)
        self.assertEqual(decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_dgfm_requires_three_global_blocks_before_deepest(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_dgfm import _resolve_dgfm_indexes

        class FakeBlock(torch.nn.Module):
            def __init__(self, window_size: int) -> None:
                super().__init__()
                self.window_size = window_size

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [FakeBlock(window_size=0), FakeBlock(window_size=14), FakeBlock(window_size=0)]
                )

        with self.assertRaisesRegex(ValueError, "at least 3 global attention blocks"):
            _resolve_dgfm_indexes(FakeImageEncoder())

    def test_unetformer_prealign_auxalign_dgsf10_uses_aligned_aux_and_dgsf_features(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.UNetFormer_MMSAM_prealign_auxalign_dgsf10 import (
                UNetFormerPreAlignAuxAlignDGSF10,
            )

        events: list[str] = []

        class FakeAuxPreAlign(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shape: tuple[int, ...] | None = None

            def forward(self, y: torch.Tensor) -> torch.Tensor:
                self.input_shape = tuple(y.shape)
                return y.repeat(1, 3, 1, 1)

        class FakePatchEmbed(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.output_rgb = x.detach().clone() + self.rgb_offset
                self.output_aux = y.detach().clone() + self.aux_offset
                return x + self.rgb_offset, y + self.aux_offset

        class ForbiddenNeck(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                del x
                raise AssertionError("DGSF10 features must be captured before encoder.neck")

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=14, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = ForbiddenNeck()

        class SpyDGSF10(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rgb_feats: tuple[torch.Tensor, ...] | None = None
                self.aux_feats: tuple[torch.Tensor, ...] | None = None
                self.outputs = tuple(torch.full((2, 4, 2, 2), float(index)) for index in range(1, 5))

            def forward(
                self,
                rgb_feats: tuple[torch.Tensor, ...],
                aux_feats: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                events.append("dgsf10")
                self.rgb_feats = tuple(feature.detach().clone() for feature in rgb_feats)
                self.aux_feats = tuple(feature.detach().clone() for feature in aux_feats)
                return self.outputs

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1, res2, res3, res4)
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        def make_model() -> torch.nn.Module:
            model = UNetFormerPreAlignAuxAlignDGSF10.__new__(UNetFormerPreAlignAuxAlignDGSF10)
            torch.nn.Module.__init__(model)
            model.aux_prealign = FakeAuxPreAlign()
            model.image_encoder = FakeImageEncoder()
            model.align_index = 2
            model.dgsf10_indexes = [0, 2, 3, 4]
            model.dgsf10 = SpyDGSF10()
            model.decoder = SpyDecoder()
            return model

        model = make_model()
        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8), return_align=False)

        dgsf10 = model.dgsf10
        decoder = model.decoder
        assert isinstance(dgsf10, SpyDGSF10)
        assert isinstance(decoder, SpyDecoder)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))
        self.assertEqual(model.aux_prealign.input_shape, (2, 1, 8, 8))
        self.assertEqual(events, ["block0", "block1", "block2", "block3", "block4", "block5", "dgsf10", "decoder"])
        self.assertIsNotNone(dgsf10.rgb_feats)
        self.assertIsNotNone(dgsf10.aux_feats)
        for feature_index, block_index in enumerate(model.dgsf10_indexes):
            block = model.image_encoder.blocks[block_index]
            assert isinstance(block, FakeBlock)
            self.assertTrue(torch.equal(dgsf10.rgb_feats[feature_index], block.output_rgb.permute(0, 3, 1, 2)))
            self.assertTrue(torch.equal(dgsf10.aux_feats[feature_index], block.output_aux.permute(0, 3, 1, 2)))
        final_block = model.image_encoder.blocks[-1]
        assert isinstance(final_block, FakeBlock)
        self.assertTrue(torch.equal(dgsf10.rgb_feats[4], final_block.output_rgb.permute(0, 3, 1, 2)))
        self.assertTrue(torch.equal(dgsf10.aux_feats[4], final_block.output_aux.permute(0, 3, 1, 2)))
        self.assertIsNotNone(decoder.inputs)
        for actual, expected in zip(decoder.inputs, dgsf10.outputs):
            self.assertIs(actual, expected)
        self.assertEqual(decoder.output_size, (8, 8))

        events.clear()
        model = make_model()
        logits, x_align_feat, y_align_feat = model(
            torch.zeros(2, 3, 8, 8),
            torch.ones(2, 8, 8),
            return_align=True,
        )

        self.assertEqual(tuple(logits.shape), (2, 6, 8, 8))
        self.assertEqual(x_align_feat.shape, y_align_feat.shape)
        align_block = model.image_encoder.blocks[2]
        assert isinstance(align_block, FakeBlock)
        self.assertTrue(torch.equal(x_align_feat, align_block.output_rgb))
        self.assertTrue(torch.equal(y_align_feat, align_block.output_aux))

    def test_unetformer_prealign_dga_applies_dga_after_all_global_blocks(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_prealign_dga10 import UNetFormerPreAlignDGA10
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        events: list[str] = []

        class FakeAuxPreAlign(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shape: tuple[int, ...] | None = None

            def forward(self, y: torch.Tensor) -> torch.Tensor:
                self.input_shape = tuple(y.shape)
                return y.repeat(1, 3, 1, 1)

        class FakePatchEmbed(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.input_rgb: torch.Tensor | None = None
                self.input_aux: torch.Tensor | None = None

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.input_rgb = x.detach().clone()
                self.input_aux = y.detach().clone()
                return x + self.rgb_offset, y + self.aux_offset

        class FakeNeck(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append("neck")
                self.inputs.append(x.detach().clone())
                return x

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=14, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                        FakeBlock("block5", window_size=0, rgb_offset=6.0, aux_offset=60.0),
                    ]
                )
                self.neck = FakeNeck()

        class FakeFPN(torch.nn.Module):
            def __init__(self, name: str, offset: float) -> None:
                super().__init__()
                self.name = name
                self.offset = offset
                self.last_output: torch.Tensor | None = None

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                self.last_output = x + self.offset
                return self.last_output

        class SpyDGA(torch.nn.Module):
            def __init__(self, name: str, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                self.output_rgb = rgb.detach().clone() + self.rgb_offset
                self.output_aux = aux.detach().clone() + self.aux_offset
                return rgb + self.rgb_offset, aux + self.aux_offset

        class SpyFusion(torch.nn.Module):
            def __init__(self, name: str) -> None:
                super().__init__()
                self.name = name
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                return rgb + aux

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                self.output_size: tuple[int, int] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (
                    res1.detach().clone(),
                    res2.detach().clone(),
                    res3.detach().clone(),
                    res4.detach().clone(),
                )
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        model = UNetFormerPreAlignDGA10.__new__(UNetFormerPreAlignDGA10)
        torch.nn.Module.__init__(model)
        model.aux_prealign = FakeAuxPreAlign()
        model.image_encoder = FakeImageEncoder()
        model.dga_indexes = [0, 2, 4, 5]
        model.dga_blocks = torch.nn.ModuleList(
            [
                SpyDGA("dga10_0", rgb_offset=100.0, aux_offset=1000.0),
                SpyDGA("dga10_1", rgb_offset=200.0, aux_offset=2000.0),
                SpyDGA("dga10_2", rgb_offset=300.0, aux_offset=3000.0),
                SpyDGA("dga10_3", rgb_offset=400.0, aux_offset=4000.0),
            ]
        )
        model.fpn1x = FakeFPN("fpn1x", 1.0)
        model.fpn2x = FakeFPN("fpn2x", 2.0)
        model.fpn3x = FakeFPN("fpn3x", 3.0)
        model.fpn4x = FakeFPN("fpn4x", 4.0)
        model.fpn1y = FakeFPN("fpn1y", 10.0)
        model.fpn2y = FakeFPN("fpn2y", 20.0)
        model.fpn3y = FakeFPN("fpn3y", 30.0)
        model.fpn4y = FakeFPN("fpn4y", 40.0)
        model.fusion1 = SpyFusion("fusion1")
        model.fusion2 = SpyFusion("fusion2")
        model.fusion3 = SpyFusion("fusion3")
        model.fusion4 = SpyFusion("fusion4")
        model.decoder = SpyDecoder()

        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8))

        self.assertEqual(model.aux_prealign.input_shape, (2, 1, 8, 8))
        self.assertEqual([len(dga_block.calls) for dga_block in model.dga_blocks], [1, 1, 1, 1])
        for dga_block in model.dga_blocks:
            self.assertEqual(dga_block.calls[0][0].shape, (2, 4, 2, 2))
            self.assertEqual(dga_block.calls[0][1].shape, (2, 4, 2, 2))

        expected_order = [
            "block0",
            "dga10_0",
            "block1",
            "block2",
            "dga10_1",
            "block3",
            "block4",
            "dga10_2",
            "block5",
            "dga10_3",
        ]
        self.assertEqual(events[: len(expected_order)], expected_order)
        self.assertLess(events.index("dga10_3"), events.index("neck"))

        block1 = model.image_encoder.blocks[1]
        block3 = model.image_encoder.blocks[3]
        block5 = model.image_encoder.blocks[5]
        self.assertTrue(torch.equal(block1.input_rgb, model.dga_blocks[0].output_rgb.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block1.input_aux, model.dga_blocks[0].output_aux.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block3.input_rgb, model.dga_blocks[1].output_rgb.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block3.input_aux, model.dga_blocks[1].output_aux.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block5.input_rgb, model.dga_blocks[2].output_rgb.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(block5.input_aux, model.dga_blocks[2].output_aux.permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(model.image_encoder.neck.inputs[0], model.dga_blocks[3].output_rgb))
        self.assertTrue(torch.equal(model.image_encoder.neck.inputs[1], model.dga_blocks[3].output_aux))

        self.assertLess(events.index("fusion1"), events.index("decoder"))
        self.assertLess(events.index("fusion2"), events.index("decoder"))
        self.assertLess(events.index("fusion3"), events.index("decoder"))
        self.assertLess(events.index("fusion4"), events.index("decoder"))
        self.assertIsNotNone(model.fpn1x.last_output)
        self.assertIsNotNone(model.fpn1y.last_output)
        self.assertTrue(torch.equal(model.fusion1.calls[0][0], model.fpn1x.last_output))
        self.assertTrue(torch.equal(model.fusion1.calls[0][1], model.fpn1y.last_output))
        self.assertEqual(model.decoder.output_size, (8, 8))
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))

    def test_unetformer_prealign_dga_requires_four_global_blocks(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_prealign_dga10 import UNetFormerPreAlignDGA10
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        class FakeBlock(torch.nn.Module):
            def __init__(self, window_size: int) -> None:
                super().__init__()
                self.window_size = window_size

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock(window_size=0),
                        FakeBlock(window_size=14),
                        FakeBlock(window_size=0),
                    ]
                )

        model = UNetFormerPreAlignDGA10.__new__(UNetFormerPreAlignDGA10)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()

        with self.assertRaises(ValueError):
            model._resolve_dga_indexes()

    def test_unetformer_prealign_auxalign_dga_applies_dga_and_returns_align_features(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_prealign_auxalign_dga10 import UNetFormerPreAlignAuxAlignDGA10
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        events: list[str] = []

        class FakeAuxPreAlign(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_shape: tuple[int, ...] | None = None

            def forward(self, y: torch.Tensor) -> torch.Tensor:
                self.input_shape = tuple(y.shape)
                return y.repeat(1, 3, 1, 1)

        class FakePatchEmbed(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
                x = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1)
                return x.permute(0, 2, 3, 1)

        class FakeBlock(torch.nn.Module):
            def __init__(self, name: str, window_size: int, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.window_size = window_size
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                return x + self.rgb_offset, y + self.aux_offset

        class FakeNeck(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs: list[torch.Tensor] = []

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                events.append("neck")
                self.inputs.append(x.detach().clone())
                return x

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed_dim = 4
                self.patch_embed = FakePatchEmbed()
                self.pos_embed = None
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock("block0", window_size=0, rgb_offset=1.0, aux_offset=10.0),
                        FakeBlock("block1", window_size=14, rgb_offset=2.0, aux_offset=20.0),
                        FakeBlock("block2", window_size=0, rgb_offset=3.0, aux_offset=30.0),
                        FakeBlock("block3", window_size=0, rgb_offset=4.0, aux_offset=40.0),
                        FakeBlock("block4", window_size=0, rgb_offset=5.0, aux_offset=50.0),
                    ]
                )
                self.neck = FakeNeck()

        class SpyDGA(torch.nn.Module):
            def __init__(self, name: str, rgb_offset: float, aux_offset: float) -> None:
                super().__init__()
                self.name = name
                self.rgb_offset = rgb_offset
                self.aux_offset = aux_offset
                self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []
                self.output_rgb: torch.Tensor | None = None
                self.output_aux: torch.Tensor | None = None

            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                events.append(self.name)
                self.calls.append((rgb.detach().clone(), aux.detach().clone()))
                self.output_rgb = rgb.detach().clone() + self.rgb_offset
                self.output_aux = aux.detach().clone() + self.aux_offset
                return rgb + self.rgb_offset, aux + self.aux_offset

        class SpyFusion(torch.nn.Module):
            def forward(self, rgb: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
                return rgb + aux

        class SpyDecoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.output_size: tuple[int, int] | None = None
                self.inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None

            def forward(
                self,
                res1: torch.Tensor,
                res2: torch.Tensor,
                res3: torch.Tensor,
                res4: torch.Tensor,
                h: int,
                w: int,
            ) -> torch.Tensor:
                events.append("decoder")
                self.inputs = (res1.detach().clone(), res2.detach().clone(), res3.detach().clone(), res4.detach().clone())
                self.output_size = (h, w)
                return torch.zeros(res1.shape[0], 6, h, w)

        def make_model() -> torch.nn.Module:
            model = UNetFormerPreAlignAuxAlignDGA10.__new__(UNetFormerPreAlignAuxAlignDGA10)
            torch.nn.Module.__init__(model)
            model.aux_prealign = FakeAuxPreAlign()
            model.image_encoder = FakeImageEncoder()
            model.align_index = 2
            model.dga_indexes = [0, 2, 3, 4]
            model.dga_blocks = torch.nn.ModuleList(
                [
                    SpyDGA("dga10_0", rgb_offset=100.0, aux_offset=1000.0),
                    SpyDGA("dga10_1", rgb_offset=200.0, aux_offset=2000.0),
                    SpyDGA("dga10_2", rgb_offset=300.0, aux_offset=3000.0),
                    SpyDGA("dga10_3", rgb_offset=400.0, aux_offset=4000.0),
                ]
            )
            model.fpn1x = torch.nn.Identity()
            model.fpn2x = torch.nn.Identity()
            model.fpn3x = torch.nn.Identity()
            model.fpn4x = torch.nn.Identity()
            model.fpn1y = torch.nn.Identity()
            model.fpn2y = torch.nn.Identity()
            model.fpn3y = torch.nn.Identity()
            model.fpn4y = torch.nn.Identity()
            model.fusion1 = SpyFusion()
            model.fusion2 = SpyFusion()
            model.fusion3 = SpyFusion()
            model.fusion4 = SpyFusion()
            model.decoder = SpyDecoder()
            return model

        model = make_model()
        output = model(torch.zeros(2, 3, 8, 8), torch.ones(2, 8, 8), return_align=False)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(tuple(output.shape), (2, 6, 8, 8))
        self.assertEqual(model.aux_prealign.input_shape, (2, 1, 8, 8))
        self.assertEqual([len(dga_block.calls) for dga_block in model.dga_blocks], [1, 1, 1, 1])
        self.assertEqual(
            events[:9],
            ["block0", "dga10_0", "block1", "block2", "dga10_1", "block3", "dga10_2", "block4", "dga10_3"],
        )
        self.assertLess(events.index("dga10_3"), events.index("neck"))
        self.assertTrue(torch.equal(model.image_encoder.neck.inputs[0], model.dga_blocks[3].output_rgb))
        self.assertTrue(torch.equal(model.image_encoder.neck.inputs[1], model.dga_blocks[3].output_aux))
        self.assertEqual(model.decoder.output_size, (8, 8))

        events.clear()
        model = make_model()
        logits, x_align_feat, y_align_feat = model(
            torch.zeros(2, 3, 8, 8),
            torch.ones(2, 8, 8),
            return_align=True,
        )

        self.assertEqual(tuple(logits.shape), (2, 6, 8, 8))
        self.assertEqual(x_align_feat.shape, y_align_feat.shape)
        self.assertTrue(torch.equal(x_align_feat, model.dga_blocks[1].calls[0][0].permute(0, 2, 3, 1)))
        self.assertTrue(torch.equal(y_align_feat, model.dga_blocks[1].calls[0][1].permute(0, 2, 3, 1)))

    def test_unetformer_dga_requires_four_global_blocks(self) -> None:
        fake_timm_module = types.ModuleType("timm")
        fake_timm_models_module = types.ModuleType("timm.models")
        fake_timm_layers_module = types.ModuleType("timm.models.layers")
        fake_timm_layers_module.DropPath = torch.nn.Identity
        fake_timm_layers_module.to_2tuple = lambda value: (value, value)
        fake_timm_layers_module.trunc_normal_ = lambda tensor, std=0.02: tensor
        fake_cv2_module = types.ModuleType("cv2")
        fake_cv2_module.COLORMAP_JET = 2
        fake_cv2_module.applyColorMap = lambda image, color_map: image
        fake_cv2_module.imwrite = lambda path, image: True
        original_timm_module = sys.modules.get("timm")
        original_timm_models_module = sys.modules.get("timm.models")
        original_timm_layers_module = sys.modules.get("timm.models.layers")
        original_cv2_module = sys.modules.get("cv2")

        try:
            sys.modules["timm"] = fake_timm_module
            sys.modules["timm.models"] = fake_timm_models_module
            sys.modules["timm.models.layers"] = fake_timm_layers_module
            sys.modules["cv2"] = fake_cv2_module
            from models.mfnet.UNetFormer_MMSAM_dga10 import UNetFormerDGA10
        finally:
            if original_timm_module is None:
                del sys.modules["timm"]
            else:
                sys.modules["timm"] = original_timm_module
            if original_timm_models_module is None:
                del sys.modules["timm.models"]
            else:
                sys.modules["timm.models"] = original_timm_models_module
            if original_timm_layers_module is None:
                del sys.modules["timm.models.layers"]
            else:
                sys.modules["timm.models.layers"] = original_timm_layers_module
            if original_cv2_module is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = original_cv2_module

        class FakeBlock(torch.nn.Module):
            def __init__(self, window_size: int) -> None:
                super().__init__()
                self.window_size = window_size

        class FakeImageEncoder(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [
                        FakeBlock(window_size=0),
                        FakeBlock(window_size=14),
                        FakeBlock(window_size=0),
                    ]
                )

        model = UNetFormerDGA10.__new__(UNetFormerDGA10)
        torch.nn.Module.__init__(model)
        model.image_encoder = FakeImageEncoder()

        with self.assertRaises(ValueError):
            model._resolve_dga_indexes()
