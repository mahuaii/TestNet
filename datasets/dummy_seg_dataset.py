from __future__ import annotations

import torch
from torch.utils.data import Dataset


class DummySegDataset(Dataset):
    def __init__(
        self,
        length: int = 8,
        image_size: int = 64,
        num_classes: int = 1,
        rgb_key: str = "rgb",
    ) -> None:
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes
        self.rgb_key = rgb_key

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, object]:
        rgb = torch.rand(3, self.image_size, self.image_size)
        if self.num_classes == 1:
            target = torch.randint(0, 2, (self.image_size, self.image_size), dtype=torch.float32)
        else:
            target = torch.randint(0, self.num_classes, (self.image_size, self.image_size), dtype=torch.long)
        return {
            "inputs": {self.rgb_key: rgb},
            "target": target,
            "meta": {"sample_id": index},
        }
