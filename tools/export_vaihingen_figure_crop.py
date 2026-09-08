from __future__ import annotations

import argparse
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np
import tifffile
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import build_isprs_dataset
from data.isprs_dataset import VAIHINGEN_PALETTE
from models import build_model
from utils import load_config, normalize_legacy_state_dict_keys


def sliding_starts(length: int, window: int, stride: int) -> list[int]:
    starts = list(range(0, length - window + 1, stride))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


@torch.inference_mode()
def predict_roi(
    model: torch.nn.Module,
    rgb: torch.Tensor,
    dsm: torch.Tensor,
    *,
    device: torch.device,
    window: int,
    stride: int,
    batch_size: int,
    num_classes: int,
) -> np.ndarray:
    height, width = rgb.shape[-2:]
    coords = [
        (top, left)
        for top in sliding_starts(height, window, stride)
        for left in sliding_starts(width, window, stride)
    ]
    logits_sum = np.zeros((num_classes, height, width), dtype=np.float32)

    for offset in tqdm(range(0, len(coords), batch_size), desc="Local sliding inference"):
        batch_coords = coords[offset : offset + batch_size]
        rgb_batch = torch.stack(
            [rgb[:, top : top + window, left : left + window] for top, left in batch_coords]
        ).to(device, non_blocking=True)
        dsm_batch = torch.stack(
            [dsm[top : top + window, left : left + window] for top, left in batch_coords]
        ).to(device, non_blocking=True)
        batch_logits = model(rgb_batch, dsm_batch, mode="Test").detach().cpu().numpy()
        for logits, (top, left) in zip(batch_logits, batch_coords):
            logits_sum[:, top : top + window, left : left + window] += logits

    return logits_sum.argmax(axis=0).astype(np.uint8)


def colorize_prediction(prediction: np.ndarray) -> np.ndarray:
    colored = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for class_id, color in VAIHINGEN_PALETTE.items():
        colored[prediction == class_id] = color
    return colored


def visualize_dsm(dsm: np.ndarray) -> np.ndarray:
    low, high = np.percentile(dsm, [2.0, 98.0])
    scaled = np.clip((dsm - low) / (high - low), 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tile-id", required=True)
    parser.add_argument("--top", type=int, required=True)
    parser.add_argument("--left", type=int, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_cfg = cfg["dataset"]
    dataset = build_isprs_dataset(
        "vaihingen",
        root_dir=dataset_cfg["root_dir"],
        ids=[args.tile_id],
        patch_size=dataset_cfg["patch_size"],
        samples_per_epoch=1,
        cache=True,
        augmentation=False,
        dsm_preprocessing=dataset_cfg["dsm_preprocessing"],
        split="val",
    )
    tile = dataset.get_tile(0)
    rgb_tensor = tile["inputs"]["rgb"]
    dsm_tensor = tile["inputs"]["dsm"]
    height, width = dsm_tensor.shape

    crop_bottom = args.top + args.size
    crop_right = args.left + args.size
    if args.top < 0 or args.left < 0 or crop_bottom > height or crop_right > width:
        raise ValueError(
            f"crop ({args.top}, {args.left}, {args.size}) exceeds tile size {height}x{width}"
        )

    roi_top = args.top - args.context
    roi_left = args.left - args.context
    roi_bottom = crop_bottom + args.context
    roi_right = crop_right + args.context
    if roi_top < 0 or roi_left < 0 or roi_bottom > height or roi_right > width:
        raise ValueError("requested context exceeds the source tile")

    device = torch.device(args.device)
    model = build_model(cfg["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_state = normalize_legacy_state_dict_keys(checkpoint["model"])
    # This checkpoint predates the change that fixed the structure confidence
    # multiplier at one, so its now-obsolete trainable scalar is intentionally
    # removed before strict loading.
    del model_state["structure_branch12.confidence_alphas"]
    model.load_state_dict(model_state)
    model.to(device).eval()

    prediction_roi = predict_roi(
        model,
        rgb_tensor[:, roi_top:roi_bottom, roi_left:roi_right],
        dsm_tensor[roi_top:roi_bottom, roi_left:roi_right],
        device=device,
        window=int(dataset.patch_size[0]),
        stride=args.stride,
        batch_size=args.batch_size,
        num_classes=int(cfg["model"]["num_classes"]),
    )
    inner_top = args.context
    inner_left = args.context
    prediction = prediction_roi[
        inner_top : inner_top + args.size,
        inner_left : inner_left + args.size,
    ]

    root = Path(dataset_cfg["root_dir"])
    rgb_raw = np.asarray(
        imageio.imread(root / "rgb" / f"top_mosaic_09cm_area{args.tile_id}.tif")
    )[args.top:crop_bottom, args.left:crop_right, :3]
    dsm_raw = np.asarray(
        imageio.imread(root / "dsm" / f"dsm_09cm_matching_area{args.tile_id}.tif")
    )[args.top:crop_bottom, args.left:crop_right]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"vaihingen_area{args.tile_id}_top{args.top}_left{args.left}_size{args.size}"
    imageio.imwrite(args.output_dir / f"{stem}_rgb.png", rgb_raw.astype(np.uint8))
    imageio.imwrite(args.output_dir / f"{stem}_dsm.png", visualize_dsm(dsm_raw))
    tifffile.imwrite(args.output_dir / f"{stem}_dsm_raw.tif", dsm_raw)
    imageio.imwrite(args.output_dir / f"{stem}_prediction.png", colorize_prediction(prediction))
    imageio.imwrite(args.output_dir / f"{stem}_prediction_ids.png", prediction)

    print(f"source tile: area{args.tile_id} ({height}x{width})")
    print(f"crop rows: [{args.top}, {crop_bottom}), columns: [{args.left}, {crop_right})")
    print(f"local inference ROI rows: [{roi_top}, {roi_bottom}), columns: [{roi_left}, {roi_right})")
    print(f"outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
