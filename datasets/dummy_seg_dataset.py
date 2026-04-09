from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch.utils.data import Dataset


class DummySegDataset(Dataset):
    def __init__(
        self,
        length: int = 8,
        image_size: int = 64,
        num_classes: int = 1,
        input_channels: Mapping[str, int] | None = None,
    ) -> None:
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes
        channels_cfg = input_channels or {"rgb": 3}
        self.input_channels = OrderedDict((str(k), int(v)) for k, v in channels_cfg.items())

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        inputs = {
            key: torch.rand(channels, self.image_size, self.image_size)
            for key, channels in self.input_channels.items()
        }
        if self.num_classes == 1:
            target = torch.randint(0, 2, (self.image_size, self.image_size), dtype=torch.float32)
        else:
            target = torch.randint(0, self.num_classes, (self.image_size, self.image_size), dtype=torch.long)
        sample = {
            "inputs": inputs,
            "target": target,
            "meta": {"sample_id": index},
        }
        return sample
