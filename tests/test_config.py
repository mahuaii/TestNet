from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.config import load_config


class ConfigTest(unittest.TestCase):
    def test_load_config_supports_jsonc_comments_and_trailing_commas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "train_config.jsonc"
            path.write_text(
                """
                {
                  // line comment
                  "model": {
                    "type": "mfnet_unetformer",
                    "num_classes": 6,
                  },
                  /* block comment */
                  "train": {
                    "experiment_name": "mfnet-v1",
                  },
                }
                """,
                encoding="utf-8",
            )

            cfg = load_config(str(path))

            self.assertEqual(cfg["model"]["type"], "mfnet_unetformer")
            self.assertEqual(cfg["model"]["num_classes"], 6)
            self.assertEqual(cfg["train"]["experiment_name"], "mfnet-v1")

    def test_load_config_extends_parent_with_deep_merge_and_list_replace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parent_path = root / "base.jsonc"
            child_path = root / "child.jsonc"
            parent_path.write_text(
                """
                {
                  "run_id": "base",
                  "model": {
                    "type": "mfnet_unetformer",
                    "num_classes": 6
                  },
                  "dataset": {
                    "train_ids": ["1", "2"],
                    "val_ids": ["5"]
                  },
                  "train": {
                    "max_epochs": 50,
                    "batch_size": 2
                  }
                }
                """,
                encoding="utf-8",
            )
            child_path.write_text(
                """
                {
                  "extends": "base.jsonc",
                  "run_id": "child",
                  "dataset": {
                    "train_ids": ["3"]
                  },
                  "train": {
                    "batch_size": 4,
                    "lambda_align": 0.01
                  }
                }
                """,
                encoding="utf-8",
            )

            cfg = load_config(str(child_path))

            self.assertNotIn("extends", cfg)
            self.assertEqual(cfg["run_id"], "child")
            self.assertEqual(cfg["model"]["type"], "mfnet_unetformer")
            self.assertEqual(cfg["model"]["num_classes"], 6)
            self.assertEqual(cfg["dataset"]["train_ids"], ["3"])
            self.assertEqual(cfg["dataset"]["val_ids"], ["5"])
            self.assertEqual(cfg["train"]["max_epochs"], 50)
            self.assertEqual(cfg["train"]["batch_size"], 4)
            self.assertEqual(cfg["train"]["lambda_align"], 0.01)

    def test_load_config_resolves_extends_relative_to_child_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base_dir = root / "configs"
            child_dir = base_dir / "variants"
            child_dir.mkdir(parents=True)
            parent_path = base_dir / "base.jsonc"
            child_path = child_dir / "child.jsonc"
            parent_path.write_text('{"run_id": "base", "train": {"max_epochs": 1}}\n', encoding="utf-8")
            child_path.write_text('{"extends": "../base.jsonc", "train": {"batch_size": 2}}\n', encoding="utf-8")

            cfg = load_config(str(child_path))

            self.assertEqual(cfg["run_id"], "base")
            self.assertEqual(cfg["train"]["max_epochs"], 1)
            self.assertEqual(cfg["train"]["batch_size"], 2)

    def test_load_config_supports_multilevel_extends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            grandparent_path = root / "grandparent.jsonc"
            parent_path = root / "parent.jsonc"
            child_path = root / "child.jsonc"
            grandparent_path.write_text('{"run_id": "base", "train": {"max_epochs": 50}}\n', encoding="utf-8")
            parent_path.write_text('{"extends": "grandparent.jsonc", "train": {"batch_size": 2}}\n', encoding="utf-8")
            child_path.write_text('{"extends": "parent.jsonc", "run_id": "child"}\n', encoding="utf-8")

            cfg = load_config(str(child_path))

            self.assertEqual(cfg["run_id"], "child")
            self.assertEqual(cfg["train"]["max_epochs"], 50)
            self.assertEqual(cfg["train"]["batch_size"], 2)

    def test_load_config_rejects_extends_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_path = root / "first.jsonc"
            second_path = root / "second.jsonc"
            first_path.write_text('{"extends": "second.jsonc"}\n', encoding="utf-8")
            second_path.write_text('{"extends": "first.jsonc"}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Config inheritance cycle detected"):
                load_config(str(first_path))

    def test_load_config_rejects_non_string_extends(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "child.jsonc"
            path.write_text('{"extends": ["base.jsonc"]}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Config extends must be a string path"):
                load_config(str(path))


if __name__ == "__main__":
    unittest.main()
