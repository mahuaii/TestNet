from __future__ import annotations

import copy
import importlib.util
import tempfile
import types
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from engine.mfnet_auxalign_trainer import MFNetAuxAlignTrainer
from engine.mfnet_baseline_auxalign_trainer import MFNetBaselineAuxAlignTrainer
from utils import DataUtils, IntermediateStatsRecorder, TestNetLogger

if importlib.util.find_spec("timm") is not None:
    from models.mfnet.UNetFormer_MMSAM_auxalign import UNetFormerAuxAlign
    from models.mfnet.UNetFormer_MMSAM_prealign_auxalign import UNetFormerPreAlignAuxAlign
else:
    UNetFormerAuxAlign = None
    UNetFormerPreAlignAuxAlign = None


class _SingleBatchDataset(Dataset):
    def __init__(self, sample: dict[str, object]) -> None:
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        del index
        return self.sample


class _PatchEmbed(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_shapes: list[tuple[int, ...]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.input_shapes.append(tuple(x.shape))
        x = torch.nn.functional.avg_pool2d(x, kernel_size=4)
        x = x.mean(dim=1, keepdim=True).repeat(1, 8, 1, 1)
        return x.permute(0, 2, 3, 1)


class _Block(torch.nn.Module):
    def __init__(self, window_size: int, offset: float) -> None:
        super().__init__()
        self.window_size = window_size
        self.offset = offset

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x + self.offset, y + 2.0 * self.offset


class _ImageEncoderWithAlign(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed()
        self.pos_embed = None
        self.blocks = torch.nn.ModuleList(
            [
                _Block(window_size=14, offset=1.0),
                _Block(window_size=14, offset=2.0),
                _Block(window_size=0, offset=3.0),
                _Block(window_size=14, offset=4.0),
            ]
        )
        self.neck = torch.nn.Conv2d(8, 256, kernel_size=1)
        self.forward_aux_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.forward_aux_shape = tuple(y.shape)
        feat = torch.zeros(x.shape[0], 256, max(1, x.shape[-2] // 4), max(1, x.shape[-1] // 4))
        return feat, feat


class _SpyAuxPreAlign(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_shape: tuple[int, ...] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_shape = tuple(x.shape)
        return x.repeat(1, 3, 1, 1)


class _Fusion(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        del y
        return x


class _Decoder(torch.nn.Module):
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
        return res1[:, :2].mean(dim=(-2, -1), keepdim=True).expand(res1.shape[0], 2, h, w)


class _ToyPreAlign(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _AuxAlignToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aux_prealign = _ToyPreAlign()
        self.encoder_adapter = torch.nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        rgb: torch.Tensor,
        dsm: torch.Tensor,
        mode: str = "Train",
        return_align: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del mode
        if not return_align:
            raise AssertionError("trainer must request align features")
        y_aligned = self.aux_prealign(dsm.unsqueeze(1))
        logits = torch.stack(
            [
                rgb[:, 0] + y_aligned[:, 0],
                rgb[:, 0] - y_aligned[:, 0],
            ],
            dim=1,
        )
        x_align = torch.ones(rgb.shape[0], 1, device=rgb.device)
        y_align = y_aligned.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1) * self.encoder_adapter
        intermediate_stats = getattr(self, "intermediate_stats", None)
        if intermediate_stats is not None:
            intermediate_stats.record_scalar("align/value", 11.0)
        return logits, x_align, y_align


class _BaselineAuxAlignToyBlock(torch.nn.Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.DSM_Adapter = torch.nn.Parameter(torch.tensor(0.25))
        self.MLPy_Adapter = torch.nn.Parameter(torch.tensor(0.5))
        self.Img_Adapter = torch.nn.Parameter(torch.tensor(0.75))
        self.MLPx_Adapter = torch.nn.Parameter(torch.tensor(1.0))
        self.wx_Adapter = torch.nn.Parameter(torch.tensor(0.5))
        self.wy_Adapter = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_next = x + self.Img_Adapter + self.MLPx_Adapter + self.wx_Adapter * y
        y_next = y + self.DSM_Adapter + self.MLPy_Adapter + self.wy_Adapter * x
        return x_next, y_next


class _BaselineAuxAlignToyImageEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [
                _BaselineAuxAlignToyBlock(window_size=14),
                _BaselineAuxAlignToyBlock(window_size=0),
                _BaselineAuxAlignToyBlock(window_size=14),
            ]
        )


class _BaselineAuxAlignToyModel(torch.nn.Module):
    def __init__(self, nonfinite: str | None = None) -> None:
        super().__init__()
        self.align_index = 1
        self.nonfinite = nonfinite
        self.image_encoder = _BaselineAuxAlignToyImageEncoder()
        self.decoder = torch.nn.Linear(1, 1, bias=False)

    def forward(
        self,
        rgb: torch.Tensor,
        dsm: torch.Tensor,
        mode: str = "Train",
        return_align: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del mode
        if not return_align:
            raise AssertionError("trainer must request align features")
        x_align = rgb.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        y_align = dsm.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
        for index, block in enumerate(self.image_encoder.blocks):
            x_align, y_align = block(x_align, y_align)
            if index == self.align_index:
                captured_x_align = x_align
                captured_y_align = y_align

        if self.nonfinite == "x_align_feat":
            captured_x_align = captured_x_align * torch.tensor(float("nan"), device=rgb.device)
        if self.nonfinite == "y_align_feat":
            captured_y_align = captured_y_align * torch.tensor(float("nan"), device=rgb.device)
        if self.nonfinite == "loss_align":
            captured_y_align = captured_y_align * torch.tensor(1.0e20, device=rgb.device)

        score = self.decoder(x_align + y_align).view(-1, 1, 1)
        logits = torch.stack(
            [
                score.expand(-1, rgb.shape[-2], rgb.shape[-1]),
                -score.expand(-1, rgb.shape[-2], rgb.shape[-1]),
            ],
            dim=1,
        )
        if self.nonfinite == "logits":
            logits = logits * torch.tensor(float("nan"), device=rgb.device)
        return logits, captured_x_align, captured_y_align


class PreAlignAuxAlignModelTest(unittest.TestCase):
    def _make_model(self) -> torch.nn.Module:
        assert UNetFormerPreAlignAuxAlign is not None
        model = UNetFormerPreAlignAuxAlign.__new__(UNetFormerPreAlignAuxAlign)
        torch.nn.Module.__init__(model)
        model.aux_prealign = _SpyAuxPreAlign()
        model.image_encoder = _ImageEncoderWithAlign()
        model.align_index = model._resolve_align_index(None)
        model.fpn1x = torch.nn.Identity()
        model.fpn2x = torch.nn.Identity()
        model.fpn3x = torch.nn.Identity()
        model.fpn4x = torch.nn.Identity()
        model.fpn1y = torch.nn.Identity()
        model.fpn2y = torch.nn.Identity()
        model.fpn3y = torch.nn.Identity()
        model.fpn4y = torch.nn.Identity()
        model.fusion1 = _Fusion()
        model.fusion2 = _Fusion()
        model.fusion3 = _Fusion()
        model.fusion4 = _Fusion()
        model.decoder = _Decoder()
        return model

    @unittest.skipIf(UNetFormerPreAlignAuxAlign is None, "timm is required to import the model")
    def test_forward_without_align_returns_logits_only(self) -> None:
        model = self._make_model()

        output = model(torch.randn(2, 3, 16, 16), torch.randn(2, 16, 16), return_align=False)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (2, 2, 16, 16))

    @unittest.skipIf(UNetFormerPreAlignAuxAlign is None, "timm is required to import the model")
    def test_forward_with_align_returns_logits_and_matching_align_features(self) -> None:
        model = self._make_model()

        logits, x_align_feat, y_align_feat = model(
            torch.randn(2, 3, 16, 16),
            torch.randn(2, 16, 16),
            return_align=True,
        )

        self.assertEqual(model.align_index, 2)
        self.assertEqual(logits.shape, (2, 2, 16, 16))
        self.assertEqual(x_align_feat.shape, y_align_feat.shape)
        self.assertEqual(x_align_feat.shape, (2, 4, 4, 8))


class BaselineAuxAlignModelTest(unittest.TestCase):
    def _make_model(self) -> torch.nn.Module:
        assert UNetFormerAuxAlign is not None
        model = UNetFormerAuxAlign.__new__(UNetFormerAuxAlign)
        torch.nn.Module.__init__(model)
        model.image_encoder = _ImageEncoderWithAlign()
        model.align_index = model._resolve_align_index(None)
        model.fpn1x = torch.nn.Identity()
        model.fpn2x = torch.nn.Identity()
        model.fpn3x = torch.nn.Identity()
        model.fpn4x = torch.nn.Identity()
        model.fpn1y = torch.nn.Identity()
        model.fpn2y = torch.nn.Identity()
        model.fpn3y = torch.nn.Identity()
        model.fpn4y = torch.nn.Identity()
        model.fusion1 = _Fusion()
        model.fusion2 = _Fusion()
        model.fusion3 = _Fusion()
        model.fusion4 = _Fusion()
        model.decoder = _Decoder()
        return model

    @unittest.skipIf(UNetFormerAuxAlign is None, "timm is required to import the model")
    def test_forward_without_align_returns_logits_only_and_repeats_aux(self) -> None:
        model = self._make_model()

        output = model(torch.randn(2, 3, 16, 16), torch.randn(2, 16, 16), return_align=False)

        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (2, 2, 16, 16))
        self.assertFalse(hasattr(model, "aux_prealign"))
        self.assertEqual(model.image_encoder.forward_aux_shape, (2, 3, 16, 16))

    @unittest.skipIf(UNetFormerAuxAlign is None, "timm is required to import the model")
    def test_forward_with_align_returns_logits_and_matching_align_features(self) -> None:
        model = self._make_model()

        logits, x_align_feat, y_align_feat = model(
            torch.randn(2, 3, 16, 16),
            torch.randn(2, 16, 16),
            return_align=True,
        )

        self.assertEqual(model.align_index, 2)
        self.assertEqual(logits.shape, (2, 2, 16, 16))
        self.assertEqual(x_align_feat.shape, y_align_feat.shape)
        self.assertEqual(x_align_feat.shape, (2, 4, 4, 8))
        self.assertEqual(
            model.image_encoder.patch_embed.input_shapes,
            [(2, 3, 16, 16), (2, 3, 16, 16)],
        )


class MFNetAuxAlignTrainerTest(unittest.TestCase):
    def _build_trainer(
        self,
        tmpdir: str,
        model: torch.nn.Module,
        trainer_cls: type[MFNetAuxAlignTrainer] | type[MFNetBaselineAuxAlignTrainer] = MFNetAuxAlignTrainer,
    ) -> MFNetAuxAlignTrainer | MFNetBaselineAuxAlignTrainer:
        sample = {
            "inputs": {
                "rgb": torch.rand(3, 8, 8),
                "dsm": torch.rand(8, 8),
            },
            "target": torch.randint(0, 2, (8, 8), dtype=torch.long),
            "meta": {"tile_id": "1"},
        }
        train_loader = DataLoader(_SingleBatchDataset(sample), batch_size=1, shuffle=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        return trainer_cls(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            train_loader=train_loader,
            val_loader=[],
            logger=TestNetLogger(tmpdir, use_tensorboard=False),
            evaluator=None,
            inferencer=None,
            device=torch.device("cpu"),
            cfg={
                "work_dir": tmpdir,
                "max_epochs": 1,
                "batch_size": 1,
                "log_step_interval": 1,
                "val_epoch_interval": 0,
                "save_epoch_interval": 1,
                "save_step_interval": 0,
                "lambda_align": 0.01,
            },
        )

    def test_train_forward_adds_align_grad_only_to_prealign_params(self) -> None:
        model = _AuxAlignToyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._build_trainer(tmpdir, model)
            batch = next(iter(trainer.train_loader))

            loss, metrics = trainer.train_forward(batch)

        self.assertEqual(loss.ndim, 0)
        self.assertIn("loss", metrics)
        self.assertIn("loss_seg", metrics)
        self.assertIn("loss_align", metrics)
        self.assertIn("accuracy", metrics)
        self.assertIsNotNone(model.aux_prealign.weight.grad)
        self.assertIsNone(model.encoder_adapter.grad)

    def test_train_forward_adds_align_grad_only_to_pre_align_dsm_adapters(self) -> None:
        model = _BaselineAuxAlignToyModel()
        expected_model = copy.deepcopy(model)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._build_trainer(tmpdir, model, trainer_cls=MFNetBaselineAuxAlignTrainer)
            batch = next(iter(trainer.train_loader))
            expected_grads = self._expected_baseline_auxalign_grads(
                trainer=trainer,
                model=expected_model,
                batch=batch,
            )

            loss, metrics = trainer.train_forward(batch)

        self.assertEqual(loss.ndim, 0)
        self.assertIn("loss_align", metrics)
        actual_grads = {name: param.grad for name, param in model.named_parameters()}
        self.assertEqual(set(actual_grads), set(expected_grads))
        for name, expected_grad in expected_grads.items():
            actual_grad = actual_grads[name]
            if expected_grad is None:
                self.assertIsNone(actual_grad, name)
            else:
                self.assertIsNotNone(actual_grad, name)
                torch.testing.assert_close(actual_grad, expected_grad, msg=name)

    def test_baseline_auxalign_trainer_rejects_nonfinite_tensors(self) -> None:
        for nonfinite_name in ("logits", "x_align_feat", "y_align_feat", "loss_align"):
            model = _BaselineAuxAlignToyModel(nonfinite=nonfinite_name)
            with self.subTest(nonfinite_name=nonfinite_name):
                with tempfile.TemporaryDirectory() as tmpdir:
                    trainer = self._build_trainer(tmpdir, model, trainer_cls=MFNetBaselineAuxAlignTrainer)
                    batch = next(iter(trainer.train_loader))

                    with self.assertRaisesRegex(FloatingPointError, f"non-finite {nonfinite_name}"):
                        trainer.train_forward(batch)

    def _expected_baseline_auxalign_grads(
        self,
        *,
        trainer: MFNetBaselineAuxAlignTrainer,
        model: torch.nn.Module,
        batch: dict[str, object],
    ) -> dict[str, torch.Tensor | None]:
        rgb, dsm, target = trainer._extract_train_tensors(batch)
        output = model(rgb, dsm, mode="Train", return_align=True)
        assert isinstance(output, tuple) and len(output) == 3
        logits, x_align_feat, y_align_feat = output
        loss_seg = DataUtils.cross_entropy_filtered(
            logits=logits,
            target=target,
            weight=trainer.class_weights,
        )
        loss_align = torch.nn.functional.mse_loss(y_align_feat, x_align_feat.detach())
        aux_align_named_params = self._aux_align_named_params(model)
        align_grads = torch.autograd.grad(
            trainer.lambda_align * loss_align,
            [param for _, param in aux_align_named_params],
            retain_graph=True,
            allow_unused=False,
        )
        loss_seg.backward()

        expected_grads = {
            name: None if param.grad is None else param.grad.detach().clone()
            for name, param in model.named_parameters()
        }
        for (name, _), align_grad in zip(aux_align_named_params, align_grads):
            if expected_grads[name] is None:
                expected_grads[name] = align_grad.detach().clone()
            else:
                expected_grads[name] = expected_grads[name] + align_grad.detach()
        return expected_grads

    @staticmethod
    def _aux_align_named_params(model: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
        params: list[tuple[str, torch.nn.Parameter]] = []
        image_encoder = getattr(model, "image_encoder")
        for block_index, block in enumerate(list(image_encoder.blocks)[: int(model.align_index) + 1]):
            for name, param in block.named_parameters():
                if param.requires_grad and ("DSM_Adapter" in name or "MLPy_Adapter" in name):
                    params.append((f"image_encoder.blocks.{block_index}.{name}", param))
        return params

    def test_train_one_epoch_merges_intermediate_stats(self) -> None:
        model = _AuxAlignToyModel()
        model.intermediate_stats = IntermediateStatsRecorder()
        model.intermediate_stats.record_scalar("stale/value", 99.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = self._build_trainer(tmpdir, model)

            train_metrics = trainer.train_one_epoch()

        self.assertAlmostEqual(train_metrics["align/value"], 11.0)
        self.assertNotIn("stale/value", train_metrics)
        self.assertEqual(model.intermediate_stats.snapshot(), {})


if __name__ == "__main__":
    unittest.main()
