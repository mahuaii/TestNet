from __future__ import annotations

import types
import unittest

import torch

from multimodal_helpers import _fake_mfnet_optional_imports


class MMAdapter10FusionBlockTest(unittest.TestCase):
    def _make_block(self) -> torch.nn.Module:
        with _fake_mfnet_optional_imports():
            from models.mfnet.sam_adapted.ImageEncoder.vit.mmadapter_fusionblock import MMAdapter10FusionBlock

            args = types.SimpleNamespace(mid_dim=None)
            return MMAdapter10FusionBlock(
                args=args,
                dim=8,
                num_heads=2,
                mlp_ratio=2.0,
                qkv_bias=True,
                window_size=0,
            )

    def test_patch_wise_fusion_uses_patch_scalar_gates(self) -> None:
        block = self._make_block()
        x_msg = torch.randn(2, 3, 5, 8)
        y_msg = torch.randn(2, 3, 5, 8)

        x_fuse, y_fuse = block.fuse_adapter_messages(x_msg, y_msg)
        gate = block.MMAdapter_Fusion(torch.cat([x_msg, y_msg], dim=-1))
        gate_y_to_x = gate[..., 0:1]
        gate_x_to_y = gate[..., 1:2]

        self.assertEqual(x_fuse.shape, x_msg.shape)
        self.assertEqual(y_fuse.shape, y_msg.shape)
        self.assertEqual(gate.shape, (2, 3, 5, 2))
        self.assertEqual(gate_y_to_x.shape, (2, 3, 5, 1))
        self.assertEqual(gate_x_to_y.shape, (2, 3, 5, 1))
        self.assertTrue(torch.allclose(x_fuse, x_msg + gate_y_to_x * y_msg))
        self.assertTrue(torch.allclose(y_fuse, y_msg + gate_x_to_y * x_msg))

    def test_patch_wise_fusion_rejects_non_spatial_tokens(self) -> None:
        block = self._make_block()

        with self.assertRaisesRegex(ValueError, r"\[B, H, W, C\]"):
            block.fuse_adapter_messages(torch.randn(2, 15, 8), torch.randn(2, 15, 8))

    def test_mmadapter10_block_forward_and_parameters(self) -> None:
        block = self._make_block()
        x = torch.randn(2, 3, 5, 8)
        y = torch.randn(2, 3, 5, 8)

        out_x, out_y = block(x, y)
        named_parameters = dict(block.named_parameters())

        self.assertEqual(out_x.shape, x.shape)
        self.assertEqual(out_y.shape, y.shape)
        self.assertNotIn("wx_Adapter", named_parameters)
        self.assertNotIn("wy_Adapter", named_parameters)
        fusion_parameter_names = [
            name for name in named_parameters if name.startswith("MMAdapter_Fusion.")
        ]
        self.assertTrue(fusion_parameter_names)
        self.assertTrue(all(named_parameters[name].requires_grad for name in fusion_parameter_names))

    def test_image_encoder_uses_mmadapter10_block_when_args_select_it(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.sam_adapted.ImageEncoder.vit.mmadapter_fusionblock import MMAdapter10FusionBlock
            from models.mfnet.sam_adapted.sam.modeling.image_encoder import ImageEncoderViT

            args = types.SimpleNamespace(mod="sam_adpt", mid_dim=None, mm_adapter_block="mmadapter10")
            encoder = ImageEncoderViT(
                args=args,
                img_size=16,
                patch_size=16,
                embed_dim=8,
                depth=1,
                num_heads=2,
                out_chans=4,
                use_abs_pos=False,
                use_rel_pos=False,
            )

        self.assertIsInstance(encoder.blocks[0], MMAdapter10FusionBlock)

    def test_image_encoder_uses_default_adapter_fusion_block_without_mmadapter10(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.sam_adapted.ImageEncoder.vit.adapter_fusionblock import AdapterFusionBlock
            from models.mfnet.sam_adapted.sam.modeling.image_encoder import ImageEncoderViT

            args = types.SimpleNamespace(mod="sam_adpt", mid_dim=None)
            encoder = ImageEncoderViT(
                args=args,
                img_size=16,
                patch_size=16,
                embed_dim=8,
                depth=1,
                num_heads=2,
                out_chans=4,
                use_abs_pos=False,
                use_rel_pos=False,
            )

        self.assertIsInstance(encoder.blocks[0], AdapterFusionBlock)

    def test_unetformer_mmadapter10_passes_mmadapter10_arg_to_sam_builder(self) -> None:
        with _fake_mfnet_optional_imports():
            import models.mfnet.UNetFormer_MMSAM as module

            captured_args: list[object] = []

            class FakeSam(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.image_encoder = torch.nn.Module()

            def fake_build_sam(args: object, checkpoint: str) -> FakeSam:
                del checkpoint
                captured_args.append(args)
                return FakeSam()

            original_parse_args = module.cfg.parse_args
            original_builder = module.sam_model_registry.get("vit_b")
            try:
                module.cfg.parse_args = lambda: types.SimpleNamespace(mod="sam_adpt")
                module.sam_model_registry["vit_b"] = fake_build_sam

                module.UNetFormerMMAdapter10(
                    num_classes=6,
                    sam_backbone="vit_b",
                    sam_checkpoint="/tmp/sam_vit_b_01ec64.pth",
                )
            finally:
                module.cfg.parse_args = original_parse_args
                if original_builder is None:
                    del module.sam_model_registry["vit_b"]
                else:
                    module.sam_model_registry["vit_b"] = original_builder

        self.assertEqual(len(captured_args), 1)
        self.assertEqual(captured_args[0].mm_adapter_block, "mmadapter10")

    def test_unetformer_prealign_mmadapter10_passes_mmadapter10_arg_and_reuses_prealign_forward(self) -> None:
        with _fake_mfnet_optional_imports():
            import models.mfnet.UNetFormer_MMSAM as base_module
            from models.mfnet.UNetFormer_MMSAM_prealign import UNetFormerPreAlign
            from models.mfnet.UNetFormer_MMSAM_prealign_mmadapter10 import UNetFormerPreAlignMMAdapter10

            captured_args: list[object] = []

            class FakeSam(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.image_encoder = torch.nn.Module()

            def fake_build_sam(args: object, checkpoint: str) -> FakeSam:
                del checkpoint
                captured_args.append(args)
                return FakeSam()

            original_parse_args = base_module.cfg.parse_args
            original_builder = base_module.sam_model_registry.get("vit_b")
            try:
                base_module.cfg.parse_args = lambda: types.SimpleNamespace(mod="sam_adpt")
                base_module.sam_model_registry["vit_b"] = fake_build_sam

                UNetFormerPreAlignMMAdapter10(
                    num_classes=6,
                    sam_backbone="vit_b",
                    sam_checkpoint="/tmp/sam_vit_b_01ec64.pth",
                )
            finally:
                base_module.cfg.parse_args = original_parse_args
                if original_builder is None:
                    del base_module.sam_model_registry["vit_b"]
                else:
                    base_module.sam_model_registry["vit_b"] = original_builder

        self.assertTrue(issubclass(UNetFormerPreAlignMMAdapter10, UNetFormerPreAlign))
        self.assertNotIn("forward", UNetFormerPreAlignMMAdapter10.__dict__)
        self.assertIs(UNetFormerPreAlignMMAdapter10.forward, UNetFormerPreAlign.forward)
        self.assertEqual(len(captured_args), 1)
        self.assertEqual(captured_args[0].mm_adapter_block, "mmadapter10")

class MMAdapter20FusionBlockTest(unittest.TestCase):
    def _make_block(self) -> torch.nn.Module:
        with _fake_mfnet_optional_imports():
            from models.mfnet.sam_adapted.ImageEncoder.vit.mmadapter_fusionblock import MMAdapter20FusionBlock

            args = types.SimpleNamespace(mid_dim=None)
            return MMAdapter20FusionBlock(
                args=args,
                dim=8,
                num_heads=2,
                mlp_ratio=2.0,
                qkv_bias=True,
                window_size=0,
            )

    def test_aux_guided_single_direction_gate_updates_rgb_only(self) -> None:
        block = self._make_block()
        x_msg = torch.randn(2, 3, 5, 8)
        y_msg = torch.randn(2, 3, 5, 8)

        x_fuse, y_fuse = block.fuse_adapter_messages(x_msg, y_msg)
        gate = torch.sigmoid(block.MMAdapter_Fusion(y_msg))

        self.assertEqual(x_fuse.shape, x_msg.shape)
        self.assertEqual(y_fuse.shape, y_msg.shape)
        self.assertEqual(gate.shape, (2, 3, 5, 1))
        self.assertTrue(torch.allclose(x_fuse, block.alpha * gate * x_msg))
        self.assertTrue(torch.equal(y_fuse, torch.zeros_like(y_msg)))

    def test_aux_guided_single_direction_gate_rejects_non_spatial_tokens(self) -> None:
        block = self._make_block()

        with self.assertRaisesRegex(ValueError, r"\[B, H, W, C\]"):
            block.fuse_adapter_messages(torch.randn(2, 15, 8), torch.randn(2, 15, 8))

    def test_mmadapter20_block_forward_and_trainable_parameters(self) -> None:
        block = self._make_block()
        x = torch.randn(2, 3, 5, 8)
        y = torch.randn(2, 3, 5, 8)

        out_x, out_y = block(x, y)
        named_parameters = dict(block.named_parameters())

        self.assertEqual(out_x.shape, x.shape)
        self.assertEqual(out_y.shape, y.shape)
        self.assertNotIn("wx_Adapter", named_parameters)
        self.assertNotIn("wy_Adapter", named_parameters)
        self.assertIn("MMAdapter_alpha", named_parameters)
        self.assertTrue(torch.allclose(block.alpha.detach(), torch.tensor(1e-3)))
        self.assertTrue(named_parameters["MMAdapter_alpha"].requires_grad)
        fusion_parameter_names = [
            name for name in named_parameters if name.startswith("MMAdapter_Fusion.")
        ]
        self.assertEqual(set(fusion_parameter_names), {"MMAdapter_Fusion.weight", "MMAdapter_Fusion.bias"})
        self.assertTrue(all(named_parameters[name].requires_grad for name in fusion_parameter_names))

    def test_image_encoder_uses_mmadapter20_block_when_args_select_it(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.sam_adapted.ImageEncoder.vit.mmadapter_fusionblock import MMAdapter20FusionBlock
            from models.mfnet.sam_adapted.sam.modeling.image_encoder import ImageEncoderViT

            args = types.SimpleNamespace(mod="sam_adpt", mid_dim=None, mm_adapter_block="mmadapter20")
            encoder = ImageEncoderViT(
                args=args,
                img_size=16,
                patch_size=16,
                embed_dim=8,
                depth=1,
                num_heads=2,
                out_chans=4,
                use_abs_pos=False,
                use_rel_pos=False,
            )

        self.assertIsInstance(encoder.blocks[0], MMAdapter20FusionBlock)

    def test_unetformer_mmadapter20_passes_mmadapter20_arg_to_sam_builder(self) -> None:
        with _fake_mfnet_optional_imports():
            import models.mfnet.UNetFormer_MMSAM as module

            captured_args: list[object] = []

            class FakeSam(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.image_encoder = torch.nn.Module()

            def fake_build_sam(args: object, checkpoint: str) -> FakeSam:
                del checkpoint
                captured_args.append(args)
                return FakeSam()

            original_parse_args = module.cfg.parse_args
            original_builder = module.sam_model_registry.get("vit_b")
            try:
                module.cfg.parse_args = lambda: types.SimpleNamespace(mod="sam_adpt")
                module.sam_model_registry["vit_b"] = fake_build_sam

                module.UNetFormerMMAdapter20(
                    num_classes=6,
                    sam_backbone="vit_b",
                    sam_checkpoint="/tmp/sam_vit_b_01ec64.pth",
                )
            finally:
                module.cfg.parse_args = original_parse_args
                if original_builder is None:
                    del module.sam_model_registry["vit_b"]
                else:
                    module.sam_model_registry["vit_b"] = original_builder

        self.assertEqual(len(captured_args), 1)
        self.assertEqual(captured_args[0].mm_adapter_block, "mmadapter20")

    def test_unetformer_prealign_mmadapter20_passes_mmadapter20_arg_and_reuses_prealign_forward(self) -> None:
        with _fake_mfnet_optional_imports():
            import models.mfnet.UNetFormer_MMSAM as base_module
            from models.mfnet.UNetFormer_MMSAM_prealign import UNetFormerPreAlign
            from models.mfnet.UNetFormer_MMSAM_prealign_mmadapter20 import UNetFormerPreAlignMMAdapter20

            captured_args: list[object] = []

            class FakeSam(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.image_encoder = torch.nn.Module()

            def fake_build_sam(args: object, checkpoint: str) -> FakeSam:
                del checkpoint
                captured_args.append(args)
                return FakeSam()

            original_parse_args = base_module.cfg.parse_args
            original_builder = base_module.sam_model_registry.get("vit_b")
            try:
                base_module.cfg.parse_args = lambda: types.SimpleNamespace(mod="sam_adpt")
                base_module.sam_model_registry["vit_b"] = fake_build_sam

                UNetFormerPreAlignMMAdapter20(
                    num_classes=6,
                    sam_backbone="vit_b",
                    sam_checkpoint="/tmp/sam_vit_b_01ec64.pth",
                )
            finally:
                base_module.cfg.parse_args = original_parse_args
                if original_builder is None:
                    del base_module.sam_model_registry["vit_b"]
                else:
                    base_module.sam_model_registry["vit_b"] = original_builder

        self.assertTrue(issubclass(UNetFormerPreAlignMMAdapter20, UNetFormerPreAlign))
        self.assertNotIn("forward", UNetFormerPreAlignMMAdapter20.__dict__)
        self.assertIs(UNetFormerPreAlignMMAdapter20.forward, UNetFormerPreAlign.forward)
        self.assertEqual(len(captured_args), 1)
        self.assertEqual(captured_args[0].mm_adapter_block, "mmadapter20")

class MMAdapter21FusionBlockTest(unittest.TestCase):
    def _make_block(self) -> torch.nn.Module:
        with _fake_mfnet_optional_imports():
            from models.mfnet.sam_adapted.ImageEncoder.vit.mmadapter_fusionblock import MMAdapter21FusionBlock

            args = types.SimpleNamespace(mid_dim=None)
            return MMAdapter21FusionBlock(
                args=args,
                dim=8,
                num_heads=2,
                mlp_ratio=2.0,
                qkv_bias=True,
                window_size=0,
            )

    def test_aux_guided_local_gate_updates_rgb_only(self) -> None:
        block = self._make_block()
        x_msg = torch.randn(2, 3, 5, 8)
        y_msg = torch.randn(2, 3, 5, 8)

        x_fuse, y_fuse = block.fuse_adapter_messages(x_msg, y_msg)
        gate = torch.sigmoid(block.MMAdapter_Fusion(y_msg))

        self.assertEqual(x_fuse.shape, x_msg.shape)
        self.assertEqual(y_fuse.shape, y_msg.shape)
        self.assertEqual(gate.shape, (2, 3, 5, 1))
        self.assertTrue(torch.allclose(x_fuse, block.alpha * gate * x_msg))
        self.assertTrue(torch.equal(y_fuse, torch.zeros_like(y_msg)))

    def test_mmadapter21_block_forward_and_trainable_parameters(self) -> None:
        block = self._make_block()
        x = torch.randn(2, 3, 5, 8)
        y = torch.randn(2, 3, 5, 8)

        out_x, out_y = block(x, y)
        named_parameters = dict(block.named_parameters())

        self.assertEqual(out_x.shape, x.shape)
        self.assertEqual(out_y.shape, y.shape)
        self.assertNotIn("wx_Adapter", named_parameters)
        self.assertNotIn("wy_Adapter", named_parameters)
        self.assertIn("MMAdapter_alpha", named_parameters)
        self.assertIn("MMAdapter_Fusion.weight", named_parameters)
        self.assertIn("MMAdapter_Fusion.bias", named_parameters)
        self.assertIn("MLPy_Adapter.local_conv.weight", named_parameters)
        self.assertIn("MLPy_Adapter.local_conv.bias", named_parameters)
        self.assertEqual(block.MLPy_Adapter.local_conv.groups, 2)
        self.assertEqual(block.MLPy_Adapter.local_conv.kernel_size, (3, 3))
        self.assertEqual(block.MLPy_Adapter.local_conv.padding, (1, 1))
        self.assertTrue(named_parameters["MLPy_Adapter.local_conv.weight"].requires_grad)

    def test_image_encoder_uses_mmadapter21_block_when_args_select_it(self) -> None:
        with _fake_mfnet_optional_imports():
            from models.mfnet.sam_adapted.ImageEncoder.vit.mmadapter_fusionblock import MMAdapter21FusionBlock
            from models.mfnet.sam_adapted.sam.modeling.image_encoder import ImageEncoderViT

            args = types.SimpleNamespace(mod="sam_adpt", mid_dim=None, mm_adapter_block="mmadapter21")
            encoder = ImageEncoderViT(
                args=args,
                img_size=16,
                patch_size=16,
                embed_dim=8,
                depth=1,
                num_heads=2,
                out_chans=4,
                use_abs_pos=False,
                use_rel_pos=False,
            )

        self.assertIsInstance(encoder.blocks[0], MMAdapter21FusionBlock)

    def test_unetformer_mmadapter21_passes_mmadapter21_arg_to_sam_builder(self) -> None:
        with _fake_mfnet_optional_imports():
            import models.mfnet.UNetFormer_MMSAM as module

            captured_args: list[object] = []

            class FakeSam(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.image_encoder = torch.nn.Module()

            def fake_build_sam(args: object, checkpoint: str) -> FakeSam:
                del checkpoint
                captured_args.append(args)
                return FakeSam()

            original_parse_args = module.cfg.parse_args
            original_builder = module.sam_model_registry.get("vit_b")
            try:
                module.cfg.parse_args = lambda: types.SimpleNamespace(mod="sam_adpt")
                module.sam_model_registry["vit_b"] = fake_build_sam

                module.UNetFormerMMAdapter21(
                    num_classes=6,
                    sam_backbone="vit_b",
                    sam_checkpoint="/tmp/sam_vit_b_01ec64.pth",
                )
            finally:
                module.cfg.parse_args = original_parse_args
                if original_builder is None:
                    del module.sam_model_registry["vit_b"]
                else:
                    module.sam_model_registry["vit_b"] = original_builder

        self.assertEqual(len(captured_args), 1)
        self.assertEqual(captured_args[0].mm_adapter_block, "mmadapter21")

    def test_unetformer_prealign_mmadapter21_passes_mmadapter21_arg_and_reuses_prealign_forward(self) -> None:
        with _fake_mfnet_optional_imports():
            import models.mfnet.UNetFormer_MMSAM as base_module
            from models.mfnet.UNetFormer_MMSAM_prealign import UNetFormerPreAlign
            from models.mfnet.UNetFormer_MMSAM_prealign_mmadapter21 import UNetFormerPreAlignMMAdapter21

            captured_args: list[object] = []

            class FakeSam(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.image_encoder = torch.nn.Module()

            def fake_build_sam(args: object, checkpoint: str) -> FakeSam:
                del checkpoint
                captured_args.append(args)
                return FakeSam()

            original_parse_args = base_module.cfg.parse_args
            original_builder = base_module.sam_model_registry.get("vit_b")
            try:
                base_module.cfg.parse_args = lambda: types.SimpleNamespace(mod="sam_adpt")
                base_module.sam_model_registry["vit_b"] = fake_build_sam

                UNetFormerPreAlignMMAdapter21(
                    num_classes=6,
                    sam_backbone="vit_b",
                    sam_checkpoint="/tmp/sam_vit_b_01ec64.pth",
                )
            finally:
                base_module.cfg.parse_args = original_parse_args
                if original_builder is None:
                    del base_module.sam_model_registry["vit_b"]
                else:
                    base_module.sam_model_registry["vit_b"] = original_builder

        self.assertTrue(issubclass(UNetFormerPreAlignMMAdapter21, UNetFormerPreAlign))
        self.assertNotIn("forward", UNetFormerPreAlignMMAdapter21.__dict__)
        self.assertIs(UNetFormerPreAlignMMAdapter21.forward, UNetFormerPreAlign.forward)
        self.assertEqual(len(captured_args), 1)
        self.assertEqual(captured_args[0].mm_adapter_block, "mmadapter21")
