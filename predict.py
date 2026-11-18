#!/usr/bin/env python3
r"""Save whole-tile ISPRS predictions for models registered in this project.

Single job (use the checkpoint's matching training configuration):
    conda activate testnet
    python predict.py --config /path/to/config.jsonc \
        --checkpoint /path/to/best_miou.pth --model-name ScSAM

Run ScSAM and baseline sequentially with their own configs and weights:
    python predict.py \
        --config /path/to/scsam.jsonc /path/to/baseline.jsonc \
        --checkpoint /path/to/scsam.pth /path/to/baseline.pth \
        --model-name ScSAM baseline

Configurations and checkpoints are paired in argument order. A single model
name can be shared across jobs, e.g. Vaihingen and Potsdam with --model-name
ScSAM. Model architecture follows config.model.type; --model-type explicitly
overrides it, as in the training entrypoints. Output names do not change the
architecture. Use the saved training config when training used such an override.

RGB and DSM preprocessing, patch sizes, and strides follow the configurations.
Tile IDs default to config.dataset.val_ids; --tile-ids selects a single job's
tiles. --stride overrides the configured stride for all jobs.
Labels are not required. Overlapping logits are summed before argmax, using
the same window coordinates as validation. Each PNG has the source tile's size
and basename. Outputs default to Hangar/ISPRS_predictions/{dataset}_{model}/.
Relative dataset/SAM paths in configurations are resolved against this project.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from data.isprs_dataset import (
    POTSDAM_PALETTE,
    POTSDAM_PRESET,
    VAIHINGEN_PALETTE,
    VAIHINGEN_PRESET,
)
from engine.sliding_window_inferencer_3090 import SlidingWindowInferencer3090
from models import build_model
from utils import DataUtils, load_config, normalize_legacy_state_dict_keys
from utils.runtime_3090 import configure_3090_runtime


ROOT = Path(__file__).resolve().parent
PRESETS = {"vaihingen": VAIHINGEN_PRESET, "potsdam": POTSDAM_PRESET}
PALETTES = {"vaihingen": VAIHINGEN_PALETTE, "potsdam": POTSDAM_PALETTE}


def project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


class PredictionTiles:
    """Input-only tile interface for the existing streaming inferencer."""

    def __init__(self, name: str, cfg: dict[str, Any], tile_ids: list[str] | None) -> None:
        self.ids = [str(tile_id) for tile_id in (cfg["val_ids"] if tile_ids is None else tile_ids)]
        if not self.ids or len(set(self.ids)) != len(self.ids):
            raise ValueError("Prediction tile IDs must be nonempty and unique")
        self.preset = PRESETS[name]
        self.root = project_path(cfg["root_dir"])
        self.window_size = tuple(map(int, cfg["patch_size"]))
        self.dsm_preprocessing = cfg["dsm_preprocessing"]
        if self.dsm_preprocessing["type"] != "similarity_enhancement":
            raise ValueError("Unsupported DSM preprocessing type")

    def paths(self, tile_id: str) -> tuple[Path, Path]:
        preset = self.preset
        return (
            self.root / preset.rgb_subdir / preset.rgb_pattern.format(tile_id=tile_id),
            self.root / preset.dsm_subdir / preset.dsm_pattern.format(tile_id=tile_id),
        )

    def get_tile(self, index: int) -> dict[str, Any]:
        tile_id = self.ids[index]
        rgb_path, dsm_path = self.paths(tile_id)
        rgb = np.asarray(imageio.imread(rgb_path), dtype=np.float32)
        rgb = (rgb[:, :, self.preset.rgb_channels] / 255.0).transpose(2, 0, 1)
        dsm_raw = imageio.imread(dsm_path)
        preprocessing = self.dsm_preprocessing
        if preprocessing["enabled"]:
            dsm = DataUtils.enhance_dsm_similarity(
                dsm_raw,
                window_size=preprocessing["window_size"],
                sigma=preprocessing["sigma"],
                lambda_weight=preprocessing["lambda_weight"],
            )
        else:
            dsm = DataUtils.normalize_dsm(dsm_raw)
        if dsm.shape != rgb.shape[-2:]:
            raise ValueError(f"RGB/DSM size mismatch for tile {tile_id}: {rgb.shape}, {dsm.shape}")
        if any(size < window for size, window in zip(dsm.shape, self.window_size)):
            raise ValueError(f"Tile {tile_id} is smaller than the configured window")
        return {
            "inputs": {
                "rgb": torch.from_numpy(rgb.copy()).float(),
                "dsm": torch.from_numpy(dsm.copy()).float(),
            },
            # The inferencer carries this field through; prediction needs no GT.
            "target": None,
            "meta": {"tile_id": tile_id},
        }


def prepare_job(
    config: Path,
    checkpoint: Path,
    model_name: str,
    model_type: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cfg = load_config(config)
    name = cfg["dataset"]["name"]
    if model_type is not None:
        cfg["model"]["type"] = model_type
    if not cfg["model"]["type"]:
        raise ValueError(f"{config} has no model.type; use a saved training config or --model-type")
    cfg["model"]["sam_checkpoint"] = str(project_path(cfg["model"]["sam_checkpoint"]))
    cfg["model"]["record_intermediate_stats"] = False
    cfg["model"]["record_intermediate_modules"] = []
    dataset = PredictionTiles(name, cfg["dataset"], args.tile_ids)
    stride = int(cfg["validation"]["stride"]) if args.stride is None else args.stride
    if len(dataset.window_size) != 2 or min(dataset.window_size) <= 0:
        raise ValueError("dataset.patch_size must contain two positive dimensions")
    if not 0 < stride <= min(dataset.window_size):
        raise ValueError("validation.stride must be positive and no larger than the window")
    num_classes = int(cfg["model"]["num_classes"])
    if not 0 < num_classes <= len(PALETTES[name]):
        raise ValueError(f"No ISPRS palette for {num_classes} classes")
    required_paths = [checkpoint, Path(cfg["model"]["sam_checkpoint"])]
    for tile_id in dataset.ids:
        required_paths.extend(dataset.paths(tile_id))
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "name": name,
        "model_name": model_name,
        "config": config,
        "checkpoint": checkpoint,
        "cfg": cfg,
        "dataset": dataset,
        "stride": stride,
    }


@torch.inference_mode()
def run_job(job: dict[str, Any], args: argparse.Namespace) -> None:
    name = job["name"]
    model_name = job["model_name"]
    cfg = job["cfg"]
    dataset = job["dataset"]
    num_classes = int(cfg["model"]["num_classes"])
    device = torch.device(args.device)
    runtime = cfg.get("runtime_3090", {})
    precision = args.precision or runtime.get("precision", "float32")
    if precision not in {"float32", "bfloat16"}:
        raise ValueError(f"Unsupported inference precision: {precision}")
    if precision == "bfloat16":
        configure_3090_runtime(device, enable_tf32=runtime.get("tf32", True))
    else:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    print(f"[{name}/{model_name}] {cfg['model']['type']} checkpoint: {job['checkpoint']}", flush=True)
    model = build_model(cfg["model"])
    checkpoint = torch.load(job["checkpoint"], map_location="cpu", weights_only=True)
    model.load_state_dict(normalize_legacy_state_dict_keys(checkpoint["model"]), strict=True)
    epoch = checkpoint["epoch"]
    del checkpoint
    model.to(device).eval()

    output_dir = args.output_root / f"{name}_{model_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = np.asarray([PALETTES[name][index] for index in range(num_classes)], dtype=np.uint8)
    inferencer = SlidingWindowInferencer3090()
    context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if precision == "bfloat16"
        else nullcontext()
    )
    saved_files = []
    with context:
        for output in inferencer.run_iter(
            model=model,
            dataset=dataset,
            device=device,
            stride=job["stride"],
            batch_size=args.batch_size,
            window_size=dataset.window_size,
            num_classes=num_classes,
            input_modals=("rgb", "dsm"),
            model_kwargs={"mode": "Test"},
        ):
            tile_id = output["meta"]["tile_id"]
            source_path, _ = dataset.paths(tile_id)
            output_path = output_dir / f"{source_path.stem}.png"
            prediction = output["pred"].numpy().astype(np.uint8)
            imageio.imwrite(output_path, palette[prediction])
            saved_files.append(output_path.name)
            print(f"[{name}/{model_name}] Saved {output_path} ({prediction.shape[0]}x{prediction.shape[1]})", flush=True)

    metadata = {
        "model": model_name,
        "model_type": cfg["model"]["type"],
        "num_classes": num_classes,
        "dataset": name,
        "config": str(job["config"]),
        "checkpoint": str(job["checkpoint"]),
        "checkpoint_epoch": epoch,
        "tile_ids": dataset.ids,
        "window_size": dataset.window_size,
        "stride": job["stride"],
        "batch_size": args.batch_size,
        "precision": precision,
        "palette": {index: PALETTES[name][index] for index in range(num_classes)},
        "files": saved_files,
    }
    (output_dir / "inference.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def expand_job_values(
    values: list[str], count: int, option: str, parser: argparse.ArgumentParser
) -> list[str]:
    if len(values) == 1:
        return values * count
    if len(values) != count:
        parser.error(f"{option} requires one value or one value per config ({count})")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, nargs="+", required=True, help="Matching training configs, in job order")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True, help="One explicit checkpoint per config")
    parser.add_argument("--model-name", nargs="+", required=True, help="Output name(s), e.g. ScSAM or baseline")
    parser.add_argument("--model-type", nargs="+", help="Explicit architecture override(s), as in train.py")
    parser.add_argument("--tile-ids", nargs="+", help="Tile IDs for a single config; defaults to dataset.val_ids")
    parser.add_argument("--stride", type=int, help="Override the sliding stride for all jobs")
    parser.add_argument("--output-root", type=Path, default=ROOT.parent / "ISPRS_predictions")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--precision", choices=("float32", "bfloat16"), help="Defaults to the training config's precision")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    count = len(args.config)
    if len(args.checkpoint) != count:
        parser.error("--config and --checkpoint must have the same number of values")
    if args.tile_ids is not None and count != 1:
        parser.error("--tile-ids requires a single config")
    model_names = expand_job_values(args.model_name, count, "--model-name", parser)
    for name in model_names:
        if not name.strip() or name in {".", ".."} or "/" in name or "\\" in name:
            parser.error("--model-name must be a nonempty directory name without path separators")
    model_types = (
        [None] * count
        if args.model_type is None
        else expand_job_values(args.model_type, count, "--model-type", parser)
    )
    args.output_root = args.output_root.expanduser().resolve()

    # Check every job before starting the first potentially long inference.
    jobs = [
        prepare_job(
            config.expanduser().resolve(),
            checkpoint.expanduser().resolve(),
            model_name,
            model_type,
            args,
        )
        for config, checkpoint, model_name, model_type in zip(
            args.config, args.checkpoint, model_names, model_types
        )
    ]
    output_names = [f"{job['name']}_{job['model_name']}" for job in jobs]
    if len(set(output_names)) != len(output_names):
        parser.error("Jobs share an output directory; use distinct --model-name values")
    for job in jobs:
        run_job(job, args)
    print(f"Predictions saved under {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
