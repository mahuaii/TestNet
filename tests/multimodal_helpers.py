from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import torch
from torch.utils.data import Dataset


@contextmanager
def _fake_mfnet_optional_imports() -> object:
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
    originals = {
        "timm": sys.modules.get("timm"),
        "timm.models": sys.modules.get("timm.models"),
        "timm.models.layers": sys.modules.get("timm.models.layers"),
        "cv2": sys.modules.get("cv2"),
    }
    try:
        sys.modules["timm"] = fake_timm_module
        sys.modules["timm.models"] = fake_timm_models_module
        sys.modules["timm.models.layers"] = fake_timm_layers_module
        sys.modules["cv2"] = fake_cv2_module
        yield
    finally:
        for module_name, original in originals.items():
            if original is None:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = original


class _SingleBatchDataset(Dataset):
    def __init__(self, batch: dict[str, object]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        del index
        return self.batch


class _ListDataset(Dataset):
    def __init__(self, batches: list[dict[str, object]]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.batches[index]


class _WholeTileDataset(Dataset):
    def __init__(self, sample: dict[str, object]) -> None:
        self.sample = sample
        self.ids = ["1"]
        self.requested_indices: list[int] = []

    def __len__(self) -> int:
        raise AssertionError("Whole-tile inference should iterate tile ids, not dataset length")

    def get_tile(self, index: int) -> dict[str, object]:
        self.requested_indices.append(index)
        return self.sample
