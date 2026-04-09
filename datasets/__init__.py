from .dummy_seg_dataset import DummySegDataset
from .vaihingen_dataset import (
    POTSDAM_TRAIN_IDS,
    POTSDAM_VAL_IDS,
    VAIHINGEN_TRAIN_IDS,
    VAIHINGEN_VAL_IDS,
    PotsdamDataset,
    VaihingenDataset,
    build_isprs_dataset,
)

__all__ = [
    "DummySegDataset",
    "POTSDAM_TRAIN_IDS",
    "POTSDAM_VAL_IDS",
    "VAIHINGEN_TRAIN_IDS",
    "VAIHINGEN_VAL_IDS",
    "PotsdamDataset",
    "VaihingenDataset",
    "build_isprs_dataset",
]
