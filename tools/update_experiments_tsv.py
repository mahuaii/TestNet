from __future__ import annotations

import argparse
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path


_EXPERIMENT_ID_PATTERN = re.compile(r"([A-Za-z0-9]{5})$")
_REQUIRED_COLUMNS = ("ID", "mIoU", "OA", "F1", "BestE", "Status")


def extract_experiment_id(work_dir: str | Path) -> str:
    match = _EXPERIMENT_ID_PATTERN.search(Path(work_dir).name)
    if match is None:
        raise ValueError(f"could not extract 5-character experiment ID from {work_dir}")
    return match.group(1)


def update_experiments_tsv(
    work_dir: str | Path,
    epoch: int,
    miou: float,
    oa: float,
    f1: float,
) -> bool:
    try:
        experiment_id = extract_experiment_id(work_dir)
    except ValueError:
        return False

    tsv_path = Path(work_dir).parent / "experiments.tsv"
    if not tsv_path.is_file():
        return False

    content = tsv_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    if not lines:
        return False

    header = lines[0].split("\t")
    if any(name not in header for name in _REQUIRED_COLUMNS):
        return False

    column_indexes = {name: header.index(name) for name in _REQUIRED_COLUMNS}
    if column_indexes["ID"] != 0:
        return False

    updated = False
    output_lines = [lines[0]]
    for line in lines[1:]:
        fields = line.split("\t")
        if fields and fields[0] == experiment_id:
            if len(fields) < len(header):
                fields.extend([""] * (len(header) - len(fields)))
            fields[column_indexes["mIoU"]] = _format_percent(miou)
            fields[column_indexes["OA"]] = _format_percent(oa)
            fields[column_indexes["F1"]] = _format_percent(f1)
            fields[column_indexes["BestE"]] = str(int(epoch))
            fields[column_indexes["Status"]] = "running"
            updated = True
        output_lines.append("\t".join(fields))

    if not updated:
        return False

    newline = "\n" if content.endswith("\n") else ""
    _atomic_write_text(tsv_path, "\n".join(output_lines) + newline)
    return True


def update_experiments_tsv_from_val_metrics(
    work_dir: str | Path,
    epoch: int,
    val_metrics: Mapping[str, object],
) -> None:
    if any(key not in val_metrics for key in ("MIoU", "accuracy", "F1Score")):
        return

    try:
        update_experiments_tsv(
            work_dir=work_dir,
            epoch=epoch,
            miou=float(val_metrics["MIoU"]),
            oa=float(val_metrics["accuracy"]),
            f1=float(val_metrics["F1Score"]),
        )
    except Exception:
        return


def _format_percent(value: float) -> str:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"metric value must be finite, got {value}")
    if scalar <= 1.0:
        scalar *= 100.0
    return f"{scalar:.2f}"


def _atomic_write_text(path: Path, text: str) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(text)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update work_dirs/experiments.tsv.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--miou", required=True, type=float)
    parser.add_argument("--oa", required=True, type=float)
    parser.add_argument("--f1", required=True, type=float)
    args = parser.parse_args()

    updated = update_experiments_tsv(
        work_dir=args.work_dir,
        epoch=args.epoch,
        miou=args.miou,
        oa=args.oa,
        f1=args.f1,
    )
    return 0 if updated else 1


if __name__ == "__main__":
    raise SystemExit(main())
