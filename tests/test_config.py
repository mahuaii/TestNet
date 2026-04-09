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


if __name__ == "__main__":
    unittest.main()
