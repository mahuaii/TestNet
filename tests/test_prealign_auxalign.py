from __future__ import annotations

import importlib.util
import tempfile
import types
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from engine.mfnet_auxalign_trainer import MFNetAuxAlignTrainer
from utils import MFNetLogger

if importlib.util.find_spec("timm") is not None:
    from models.mfnet.UNetFormer_MMSAM_prealign_auxalign import UNetFormerPreAlignAuxAlign
else:
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        return logits, x_align, y_align


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


class MFNetAuxAlignTrainerTest(unittest.TestCase):
    def _build_trainer(self, tmpdir: str, model: torch.nn.Module) -> MFNetAuxAlignTrainer:
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
        return MFNetAuxAlignTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            train_loader=train_loader,
            val_loader=[],
            logger=MFNetLogger(tmpdir, use_tensorboard=False),
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


if __name__ == "__main__":
    unittest.main()
