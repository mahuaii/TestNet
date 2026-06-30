from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.update_experiments_tsv import (
    extract_experiment_id,
    update_experiments_tsv_best_metrics,
    update_experiments_tsv_status,
)


class UpdateExperimentsTsvTest(unittest.TestCase):
    def test_extract_experiment_id_uses_final_five_characters(self) -> None:
        self.assertEqual(
            extract_experiment_id("work_dirs/vaihingen_xxx_6b855"),
            "6b855",
        )

    def test_best_metrics_update_matching_row_preserves_status_and_other_rows(self) -> None:
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

            updated = update_experiments_tsv_best_metrics(
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
                "6b855\t40\t82.76\t92.30\t90.18\t32\trunning(32/48)\t\tpython train.py\n",
            )

    def test_best_metrics_missing_row_returns_false_and_leaves_file_unchanged(self) -> None:
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

            updated = update_experiments_tsv_best_metrics(
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

            updated = update_experiments_tsv_best_metrics(
                work_dir=work_dir,
                epoch=1,
                miou=0.5,
                oa=0.8,
                f1=0.7,
            )

            self.assertFalse(updated)

    def test_best_metrics_missing_required_column_returns_false_and_leaves_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir, "work_dirs")
            work_dir = root / "vaihingen_xxx_6b855"
            work_dir.mkdir(parents=True)
            tsv_path = root / "experiments.tsv"
            original = "ID\tSeed\tmIoU\tOA\tBestE\tStatus\n6b855\t40\t82.75\t92.42\t28\trunning\n"
            tsv_path.write_text(original, encoding="utf-8")

            updated = update_experiments_tsv_best_metrics(
                work_dir=work_dir,
                epoch=32,
                miou=0.8275,
                oa=92.3043,
                f1=0.9018,
            )

            self.assertFalse(updated)
            self.assertEqual(tsv_path.read_text(encoding="utf-8"), original)

    def test_status_update_only_updates_status_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir, "work_dirs")
            work_dir = root / "vaihingen_xxx_6b855"
            work_dir.mkdir(parents=True)
            tsv_path = root / "experiments.tsv"
            tsv_path.write_text(
                "ID\tSeed\tmIoU\tOA\tF1\tBestE\tStatus\n"
                "6b855\t40\t82.75\t92.42\t90.30\t28\trunning(28/48)\n",
                encoding="utf-8",
            )

            updated = update_experiments_tsv_status(
                work_dir=work_dir,
                epoch=32,
                max_epochs=48,
            )

            self.assertTrue(updated)
            self.assertEqual(
                tsv_path.read_text(encoding="utf-8"),
                "ID\tSeed\tmIoU\tOA\tF1\tBestE\tStatus\n"
                "6b855\t40\t82.75\t92.42\t90.30\t28\trunning(32/48)\n",
            )

    def test_status_update_marks_validation_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir, "work_dirs")
            work_dir = root / "vaihingen_xxx_6b855"
            work_dir.mkdir(parents=True)
            tsv_path = root / "experiments.tsv"
            tsv_path.write_text(
                "ID\tSeed\tmIoU\tOA\tF1\tBestE\tStatus\n"
                "6b855\t40\t82.75\t92.42\t90.30\t28\trunning(32/48)\n",
                encoding="utf-8",
            )

            updated = update_experiments_tsv_status(
                work_dir=work_dir,
                epoch=32,
                max_epochs=48,
                phase="val",
            )

            self.assertTrue(updated)
            self.assertEqual(
                tsv_path.read_text(encoding="utf-8"),
                "ID\tSeed\tmIoU\tOA\tF1\tBestE\tStatus\n"
                "6b855\t40\t82.75\t92.42\t90.30\t28\trunning(32/48)(val)\n",
            )

    def test_status_missing_required_column_returns_false_and_leaves_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir, "work_dirs")
            work_dir = root / "vaihingen_xxx_6b855"
            work_dir.mkdir(parents=True)
            tsv_path = root / "experiments.tsv"
            original = "ID\tSeed\tmIoU\tOA\tF1\tBestE\n6b855\t40\t82.75\t92.42\t90.30\t28\n"
            tsv_path.write_text(original, encoding="utf-8")

            updated = update_experiments_tsv_status(
                work_dir=work_dir,
                epoch=32,
                max_epochs=48,
            )

            self.assertFalse(updated)
            self.assertEqual(tsv_path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
