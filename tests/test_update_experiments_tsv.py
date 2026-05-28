from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import tools.update_experiments_tsv as update_module
from tools.update_experiments_tsv import (
    extract_experiment_id,
    update_experiments_tsv_from_val_metrics,
    update_experiments_tsv,
)


class UpdateExperimentsTsvTest(unittest.TestCase):
    def test_extract_experiment_id_uses_final_five_characters(self) -> None:
        self.assertEqual(
            extract_experiment_id("work_dirs/vaihingen_xxx_6b855"),
            "6b855",
        )

    def test_update_matching_row_preserves_columns_and_other_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir, "work_dirs")
            work_dir = root / "vaihingen_xxx_6b855"
            work_dir.mkdir(parents=True)
            tsv_path = root / "experiments.tsv"
            tsv_path.write_text(
                "ID\tSeed\tmIoU\tOA\tF1\tBestE\tStatus\tNote\tCommand\n"
                "abcde\t80\t80.00\t91.00\t89.00\t12\tdone\tkeep\tcmd\n"
                "6b855\t40\t82.75\t92.42\t90.30\t28\trunning(32/48)\t\tpython train.py\n",
                encoding="utf-8",
            )

            updated = update_experiments_tsv(
                work_dir=work_dir,
                epoch=32,
                miou=0.82756,
                oa=92.3043,
                f1=0.9018,
            )

            self.assertTrue(updated)
            self.assertEqual(
                tsv_path.read_text(encoding="utf-8"),
                "ID\tSeed\tmIoU\tOA\tF1\tBestE\tStatus\tNote\tCommand\n"
                "abcde\t80\t80.00\t91.00\t89.00\t12\tdone\tkeep\tcmd\n"
                "6b855\t40\t82.76\t92.30\t90.18\t32\trunning\t\tpython train.py\n",
            )

    def test_missing_row_returns_false_and_leaves_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir, "work_dirs")
            work_dir = root / "vaihingen_xxx_6b855"
            work_dir.mkdir(parents=True)
            tsv_path = root / "experiments.tsv"
            original = (
                "ID\tSeed\tmIoU\tOA\tF1\tBestE\tStatus\n"
                "abcde\t80\t80.00\t91.00\t89.00\t12\tdone\n"
            )
            tsv_path.write_text(original, encoding="utf-8")

            updated = update_experiments_tsv(
                work_dir=work_dir,
                epoch=32,
                miou=0.8275,
                oa=92.3043,
                f1=0.9018,
            )

            self.assertFalse(updated)
            self.assertEqual(tsv_path.read_text(encoding="utf-8"), original)

    def test_missing_experiment_id_returns_false_without_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir, "work_dirs")
            work_dir = root / "bad"
            work_dir.mkdir(parents=True)

            updated = update_experiments_tsv(
                work_dir=work_dir,
                epoch=1,
                miou=0.5,
                oa=0.8,
                f1=0.7,
            )

            self.assertFalse(updated)

    def test_missing_required_column_returns_false_and_leaves_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir, "work_dirs")
            work_dir = root / "vaihingen_xxx_6b855"
            work_dir.mkdir(parents=True)
            tsv_path = root / "experiments.tsv"
            original = "ID\tSeed\tmIoU\tOA\tBestE\tStatus\n6b855\t40\t82.75\t92.42\t28\trunning\n"
            tsv_path.write_text(original, encoding="utf-8")

            updated = update_experiments_tsv(
                work_dir=work_dir,
                epoch=32,
                miou=0.8275,
                oa=92.3043,
                f1=0.9018,
            )

            self.assertFalse(updated)
            self.assertEqual(tsv_path.read_text(encoding="utf-8"), original)

    def test_from_val_metrics_updates_with_current_validation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir, "work_dirs")
            work_dir = root / "vaihingen_xxx_6b855"
            work_dir.mkdir(parents=True)
            tsv_path = root / "experiments.tsv"
            original = (
                "ID\tSeed\tmIoU\tOA\tF1\tBestE\tStatus\n"
                "6b855\t40\t82.75\t92.42\t90.30\t28\trunning\n"
            )
            tsv_path.write_text(original, encoding="utf-8")

            result = update_experiments_tsv_from_val_metrics(
                work_dir=work_dir,
                epoch=32,
                val_metrics={"MIoU": 0.8256, "accuracy": 92.3043, "F1Score": 0.9018},
            )

            self.assertIsNone(result)
            self.assertEqual(
                tsv_path.read_text(encoding="utf-8"),
                "ID\tSeed\tmIoU\tOA\tF1\tBestE\tStatus\n"
                "6b855\t40\t82.56\t92.30\t90.18\t32\trunning\n",
            )

    def test_from_val_metrics_ignores_missing_required_metrics(self) -> None:
        result = update_experiments_tsv_from_val_metrics(
            work_dir="work_dirs/vaihingen_xxx_6b855",
            epoch=32,
            val_metrics={"MIoU": 0.8281, "accuracy": 92.2484},
        )

        self.assertIsNone(result)

    def test_from_val_metrics_ignores_update_errors(self) -> None:
        original_update = update_module.update_experiments_tsv

        def fake_update_experiments_tsv(**kwargs: object) -> bool:
            del kwargs
            raise RuntimeError("boom")

        update_module.update_experiments_tsv = fake_update_experiments_tsv
        try:
            result = update_experiments_tsv_from_val_metrics(
                work_dir="work_dirs/vaihingen_xxx_6b855",
                epoch=32,
                val_metrics={"MIoU": 0.8281, "accuracy": 92.2484, "F1Score": 0.9034},
            )
        finally:
            update_module.update_experiments_tsv = original_update

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
