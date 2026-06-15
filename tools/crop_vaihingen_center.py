from __future__ import annotations

import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tifffile

_AREA_PATTERN = re.compile(r"area(?P<area_id>\d+)")
_MODALITY_PATTERNS = {
    "rgb": "top_mosaic_09cm_area{area_id}.tif",
    "dsm": "dsm_09cm_matching_area{area_id}.tif",
    "labels": "top_mosaic_09cm_area{area_id}.tif",
    "labels_eroded": "top_mosaic_09cm_area{area_id}_noBoundary.tif",
}


@dataclass(frozen=True)
class CropTask:
    modality: str
    source: Path
    target: Path
    top: int
    left: int
    size: int


def parse_area_id(path: str | Path) -> int:
    match = _AREA_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"could not parse area ID from {path}")
    return int(match.group("area_id"))


def output_name(modality: str, area_id: int, offset: int = 100) -> str:
    try:
        pattern = _MODALITY_PATTERNS[modality]
    except KeyError as exc:
        raise ValueError(f"unsupported modality: {modality}") from exc
    return pattern.format(area_id=area_id + offset)


def center_crop_origin(height: int, width: int, size: int) -> tuple[int, int]:
    if size <= 0:
        raise ValueError(f"crop size must be positive, got {size}")
    if height < size or width < size:
        raise ValueError(f"image size {width}x{height} is smaller than crop size {size}x{size}")
    return (height - size) // 2, (width - size) // 2


def discover_common_area_ids(root: str | Path, source_id_limit: int = 50) -> list[int]:
    root_path = Path(root)
    modality_ids: list[set[int]] = []
    for modality in _MODALITY_PATTERNS:
        directory = root_path / modality
        if not directory.is_dir():
            raise FileNotFoundError(f"missing Vaihingen modality directory: {directory}")
        ids: set[int] = set()
        for path in directory.glob("*.tif"):
            area_id = parse_area_id(path)
            if area_id < source_id_limit:
                ids.add(area_id)
        modality_ids.append(ids)
    return sorted(set.intersection(*modality_ids))


def build_crop_tasks(
    root: str | Path,
    crop_size: int = 1024,
    offset: int = 100,
    source_id_limit: int = 50,
) -> list[CropTask]:
    root_path = Path(root)
    area_ids = discover_common_area_ids(root_path, source_id_limit=source_id_limit)
    if not area_ids:
        raise ValueError(f"no common source areas found under {root_path}")

    tasks: list[CropTask] = []
    for area_id in area_ids:
        shapes: dict[str, tuple[int, int]] = {}
        sources: dict[str, Path] = {}
        for modality, pattern in _MODALITY_PATTERNS.items():
            source = root_path / modality / pattern.format(area_id=area_id)
            if not source.is_file():
                raise FileNotFoundError(f"missing source file: {source}")
            with tifffile.TiffFile(source) as tiff:
                shape = tiff.series[0].shape
            if len(shape) < 2:
                raise ValueError(f"unsupported TIFF shape {shape} in {source}")
            shapes[modality] = (int(shape[0]), int(shape[1]))
            sources[modality] = source

        unique_shapes = set(shapes.values())
        if len(unique_shapes) != 1:
            details = ", ".join(f"{name}={shape}" for name, shape in shapes.items())
            raise ValueError(f"area {area_id} modality dimensions do not match: {details}")

        height, width = unique_shapes.pop()
        top, left = center_crop_origin(height, width, crop_size)
        for modality, source in sources.items():
            tasks.append(
                CropTask(
                    modality=modality,
                    source=source,
                    target=root_path / modality / output_name(modality, area_id, offset),
                    top=top,
                    left=left,
                    size=crop_size,
                )
            )
    return tasks


def _atomic_write_tiff(path: Path, image: object) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=path.suffix,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        tifffile.imwrite(temp_path, image)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def execute_crop_tasks(tasks: list[CropTask]) -> None:
    for task in tasks:
        image = tifffile.imread(task.source)
        cropped = image[
            task.top : task.top + task.size,
            task.left : task.left + task.size,
            ...,
        ]
        expected_shape = (task.size, task.size)
        if cropped.shape[:2] != expected_shape:
            raise ValueError(
                f"unexpected crop shape {cropped.shape} from {task.source}; "
                f"expected leading dimensions {expected_shape}"
            )
        _atomic_write_tiff(task.target, cropped)


def crop_vaihingen_center(
    root: str | Path,
    crop_size: int = 1024,
    offset: int = 100,
    source_id_limit: int = 50,
) -> list[CropTask]:
    tasks = build_crop_tasks(
        root=root,
        crop_size=crop_size,
        offset=offset,
        source_id_limit=source_id_limit,
    )
    execute_crop_tasks(tasks)
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create center-cropped Vaihingen areas in the existing modality directories."
    )
    parser.add_argument("--root", default="datasets/vaihingen")
    parser.add_argument("--crop-size", type=int, default=260)
    parser.add_argument("--offset", type=int, default=100)
    args = parser.parse_args()

    tasks = crop_vaihingen_center(
        root=args.root,
        crop_size=args.crop_size,
        offset=args.offset,
    )
    area_count = len(tasks) // len(_MODALITY_PATTERNS)
    print(f"generated {len(tasks)} files for {area_count} areas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
