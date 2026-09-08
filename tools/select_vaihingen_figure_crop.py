from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.isprs_dataset import VAIHINGEN_INVERT_PALETTE, VAIHINGEN_PALETTE
from utils import DataUtils


def label_ids(path: Path) -> np.ndarray:
    return DataUtils.convert_from_color(
        np.asarray(imageio.imread(path)),
        invert_palette=VAIHINGEN_INVERT_PALETTE,
    )


def colorize(labels: np.ndarray) -> np.ndarray:
    result = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for class_id, color in VAIHINGEN_PALETTE.items():
        result[labels == class_id] = color
    return result


def score_crop(labels: np.ndarray) -> float:
    counts = np.bincount(labels.ravel(), minlength=7)[:6].astype(np.float64)
    fractions = counts / counts.sum()
    active = fractions[fractions > 0.01]
    entropy = float(-(active * np.log(active)).sum())
    boundary = np.count_nonzero(labels[1:] != labels[:-1])
    boundary += np.count_nonzero(labels[:, 1:] != labels[:, :-1])
    boundary_density = boundary / (2.0 * labels.size)
    car_bonus = min(fractions[4] / 0.015, 1.0)
    class_bonus = np.count_nonzero(fractions[:5] > 0.02)
    dominance_penalty = max(0.0, float(fractions.max()) - 0.65)
    return entropy + 2.0 * boundary_density + 0.35 * car_bonus + 0.10 * class_bonus - dominance_penalty


def select_candidates(root: Path, tile_ids: list[str], size: int, step: int, count: int):
    candidates = []
    for tile_id in tile_ids:
        label_path = root / "labels" / f"top_mosaic_09cm_area{tile_id}.tif"
        labels = label_ids(label_path)
        height, width = labels.shape
        for top in range(0, height - size + 1, step):
            for left in range(0, width - size + 1, step):
                patch = labels[top : top + size, left : left + size]
                score = score_crop(patch)
                candidates.append((score, tile_id, top, left))
    candidates.sort(reverse=True)

    selected = []
    for item in candidates:
        _, tile_id, top, left = item
        if all(
            tile_id != old_tile
            or math.hypot(top - old_top, left - old_left) >= size
            for _, old_tile, old_top, old_left in selected
        ):
            selected.append(item)
            if len(selected) == count:
                break
    return selected


def create_contact_sheet(root: Path, candidates, size: int, output: Path) -> None:
    thumb = 192
    label_height = 32
    columns = 4
    rows = math.ceil(len(candidates) / columns)
    sheet = Image.new("RGB", (columns * thumb * 2, rows * (thumb + label_height)), "white")
    draw = ImageDraw.Draw(sheet)

    for index, (score, tile_id, top, left) in enumerate(candidates):
        rgb_path = root / "rgb" / f"top_mosaic_09cm_area{tile_id}.tif"
        label_path = root / "labels" / f"top_mosaic_09cm_area{tile_id}.tif"
        rgb = np.asarray(imageio.imread(rgb_path))[top : top + size, left : left + size, :3]
        labels = label_ids(label_path)[top : top + size, left : left + size]
        rgb_image = Image.fromarray(rgb.astype(np.uint8)).resize((thumb, thumb), Image.Resampling.LANCZOS)
        label_image = Image.fromarray(colorize(labels)).resize((thumb, thumb), Image.Resampling.NEAREST)
        row, col = divmod(index, columns)
        x = col * thumb * 2
        y = row * (thumb + label_height)
        sheet.paste(rgb_image, (x, y))
        sheet.paste(label_image, (x + thumb, y))
        draw.text((x + 4, y + thumb + 7), f"#{index + 1} area{tile_id} top={top} left={left} score={score:.3f}", fill="black")

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("datasets/vaihingen"))
    parser.add_argument("--tile-ids", nargs="+", default=["5", "21", "15", "30"])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--step", type=int, default=64)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("figures/vaihingen_crop_candidates.png"))
    args = parser.parse_args()

    candidates = select_candidates(args.root, args.tile_ids, args.size, args.step, args.count)
    for index, item in enumerate(candidates, 1):
        score, tile_id, top, left = item
        print(f"{index:02d}: area={tile_id} top={top} left={left} size={args.size} score={score:.6f}")
    create_contact_sheet(args.root, candidates, args.size, args.output)


if __name__ == "__main__":
    main()
