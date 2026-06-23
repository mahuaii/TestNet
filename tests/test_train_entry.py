from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from utils import TestNetLogger as _TestNetLogger
from utils import TestNetRecorderLogger as _TestNetRecorderLogger


class TrainEntryTest(unittest.TestCase):
    def _load_train_entry_module(self):
        spec = importlib.util.spec_from_file_location(
            "test_train_module",
            Path(__file__).resolve().parents[1] / "train.py",
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _make_train_entry_config(self, root_dir: str) -> dict[str, object]:
        return {
            "model": {
                "type": "mfnet_unetformer",
                "num_classes": 6,
                "sam_backbone": "vit_b",
                "sam_checkpoint": "/tmp/sam_vit_b.pth",
            },
            "dataset": {
                "name": "vaihingen",
                "root_dir": root_dir,
                "patch_size": [32, 32],
                "train_ids": ["1"],
                "train_tile_sampling_weights": [2.5],
                "val_ids": ["5"],
                "train_samples_per_epoch": 4,
                "val_samples_per_epoch": 1,
                "cache": True,
                "augmentation": True,
                "dsm_preprocessing": None,
                "patch_sampling": {"enabled": False},
            },
            "dataloader": {"num_workers": 0},
            "optimizer": {
                "type": "SGD",
                "lr": 0.01,
                "adapter_lr": 0.001,
                "momentum": 0.9,
                "weight_decay": 0.0005,
            },
            "scheduler": {"type": "MultiStepLR", "milestones": [1, 2], "gamma": 0.1},
            "validation": {
                "stride": 32,
            },
            "train": {
                "max_epochs": 1,
                "batch_size": 2,
                "auto_resume": True,
                "log_step_interval": 1,
                "val_epoch_interval": 1,
                "save_epoch_interval": 1,
                "save_step_interval": 0,
                "use_tensorboard": True,
            },
        }

    def _run_train_entry(
        self,
        module,
        args: object,
        cfg: dict[str, object] | None,
        default_work_dir: Path | None,
    ) -> dict[str, object]:
        class FakeModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.image_encoder = torch.nn.Linear(1, 1, bias=False)
                self.align_index = 2

        captured_dataset_calls: list[dict[str, object]] = []
        captured_trainer_kwargs: list[dict[str, object]] = []
        captured_trainer_classes: list[str] = []
        captured_model_cfg: list[dict[str, object]] = []
        captured_load_config_paths: list[str] = []
        captured_default_work_dir_calls: list[dict[str, object]] = []

        class FakeTrainer:
            def __init__(self, **kwargs: object) -> None:
                captured_trainer_classes.append("MFNetTrainer")
                captured_trainer_kwargs.append(kwargs)

            def train(self) -> None:
                return None

        class FakeDGATrainer:
            def __init__(self, **kwargs: object) -> None:
                captured_trainer_classes.append("MFNetDGATrainer")
                captured_trainer_kwargs.append(kwargs)

            def train(self) -> None:
                return None

        class FakeAuxAlignTrainer:
            def __init__(self, **kwargs: object) -> None:
                captured_trainer_classes.append("MFNetAuxAlignTrainer")
                captured_trainer_kwargs.append(kwargs)

            def train(self) -> None:
                return None

        class FakeBaselineAuxAlignTrainer:
            def __init__(self, **kwargs: object) -> None:
                captured_trainer_classes.append("MFNetBaselineAuxAlignTrainer")
                captured_trainer_kwargs.append(kwargs)

            def train(self) -> None:
                return None

        class FakeAuxAlignDGATrainer:
            def __init__(self, **kwargs: object) -> None:
                captured_trainer_classes.append("MFNetAuxAlignDGATrainer")
                captured_trainer_kwargs.append(kwargs)

            def train(self) -> None:
                return None

        original_parse_args = module.parse_args
        original_load_config = module.load_config
        original_build_default_work_dir = module.build_default_work_dir
        original_build_model = module.build_model
        original_build_isprs_dataset = module.build_isprs_dataset
        original_dataloader = module.DataLoader
        original_trainer = module.MFNetTrainer
        original_dga_trainer = module.MFNetDGATrainer
        original_auxalign_trainer = module.MFNetAuxAlignTrainer
        original_baseline_auxalign_trainer = module.MFNetBaselineAuxAlignTrainer
        original_auxalign_dga_trainer = module.MFNetAuxAlignDGATrainer
        try:
            module.parse_args = lambda: args

            def fake_load_config(path: str) -> dict[str, object]:
                captured_load_config_paths.append(path)
                if cfg is None:
                    return original_load_config(path)
                return cfg

            module.load_config = fake_load_config

            def fake_build_default_work_dir(
                model_name: object,
                dataset_name: object,
                lambda_align: object | None = None,
                root_dir: str | Path = "work_dirs",
            ) -> Path:
                captured_default_work_dir_calls.append(
                    {
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "lambda_align": lambda_align,
                        "root_dir": root_dir,
                    }
                )
                if default_work_dir is None:
                    raise AssertionError("resume-dir should not build a new workdir")
                return default_work_dir

            module.build_default_work_dir = fake_build_default_work_dir

            def fake_build_model(model_cfg: dict[str, object]) -> FakeModel:
                captured_model_cfg.append(model_cfg)
                return FakeModel()

            module.build_model = fake_build_model

            def fake_build_isprs_dataset(name: str, **kwargs: object) -> dict[str, object]:
                captured_dataset_calls.append({"name": name, **kwargs})
                return {"dataset_name": name, "dataset_kwargs": kwargs}

            module.build_isprs_dataset = fake_build_isprs_dataset
            module.DataLoader = lambda dataset, **kwargs: {
                "dataset": dataset,
                "batch_size": kwargs["batch_size"],
                "loader_kwargs": kwargs,
            }
            module.MFNetTrainer = FakeTrainer
            module.MFNetDGATrainer = FakeDGATrainer
            module.MFNetAuxAlignTrainer = FakeAuxAlignTrainer
            module.MFNetBaselineAuxAlignTrainer = FakeBaselineAuxAlignTrainer
            module.MFNetAuxAlignDGATrainer = FakeAuxAlignDGATrainer

            module.main()
        finally:
            module.parse_args = original_parse_args
            module.load_config = original_load_config
            module.build_default_work_dir = original_build_default_work_dir
            module.build_model = original_build_model
            module.build_isprs_dataset = original_build_isprs_dataset
            module.DataLoader = original_dataloader
            module.MFNetTrainer = original_trainer
            module.MFNetDGATrainer = original_dga_trainer
            module.MFNetAuxAlignTrainer = original_auxalign_trainer
            module.MFNetBaselineAuxAlignTrainer = original_baseline_auxalign_trainer
            module.MFNetAuxAlignDGATrainer = original_auxalign_dga_trainer

        return {
            "dataset_calls": captured_dataset_calls,
            "trainer_kwargs": captured_trainer_kwargs,
            "trainer_classes": captured_trainer_classes,
            "model_cfg": captured_model_cfg,
            "load_config_paths": captured_load_config_paths,
            "default_work_dir_calls": captured_default_work_dir_calls,
        }

    def test_train_default_work_dir_uses_model_dataset_and_short_run_id(self) -> None:
        module = self._load_train_entry_module()

        with patch.object(sys, "argv", ["train.py"]):
            args = module.parse_args()

        self.assertFalse(hasattr(args, "work_dir"))
        self.assertFalse(hasattr(args, "resume_from"))
        self.assertIsNone(args.resume_dir)
        self.assertIsNone(args.resume_ckpt)
        self.assertIsNone(args.model_type)

        with patch.object(sys, "argv", ["train.py", "--model-type", "testnet_dga10"]):
            args = module.parse_args()

        self.assertEqual(args.model_type, "testnet_dga10")

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = module.build_default_work_dir(
                model_name="MFNet UNetFormer",
                dataset_name="Potsdam/RGB",
                root_dir=tmpdir,
            )

            self.assertEqual(work_dir.parent, Path(tmpdir))
            self.assertRegex(
                work_dir.name,
                re.compile(r"^potsdam_rgb_base_[0-9a-f]{5}$"),
            )

            dga_work_dir = module.build_default_work_dir(
                model_name="testnet_dga20",
                dataset_name="vaihingen",
                root_dir=tmpdir,
            )
            self.assertRegex(
                dga_work_dir.name,
                re.compile(r"^vaihingen_dga20_[0-9a-f]{5}$"),
            )

            lambda_work_dir = module.build_default_work_dir(
                model_name="testnet_prealign_auxalign",
                dataset_name="vaihingen",
                lambda_align=0.01,
                root_dir=tmpdir,
            )
            self.assertRegex(
                lambda_work_dir.name,
                re.compile(r"^vaihingen_prealign_auxalign_lambda-0.01_[0-9a-f]{5}$"),
            )

    def test_optimizer_param_groups_exclude_gate_scalars_from_weight_decay(self) -> None:
        module = self._load_train_entry_module()

        class FakeGateModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2, 2))
                self.bias = torch.nn.Parameter(torch.ones(2))
                self.alpha = torch.nn.Parameter(torch.ones(1))
                self.beta = torch.nn.Parameter(torch.ones(1))
                self.gamma = torch.nn.Parameter(torch.ones(1))
                self.lora_alpha = torch.nn.Parameter(torch.ones(2, 2))
                self.register_parameter("lambda", torch.nn.Parameter(torch.ones(1)))

        model = FakeGateModel()
        param_groups = module.build_optimizer_param_groups(model, weight_decay=0.0005)

        self.assertEqual(len(param_groups), 2)
        decay_group = next(group for group in param_groups if group["weight_decay"] == 0.0005)
        no_decay_group = next(group for group in param_groups if group["weight_decay"] == 0.0)
        decay_param_ids = {id(param) for param in decay_group["params"]}
        no_decay_param_ids = {id(param) for param in no_decay_group["params"]}

        self.assertIn(id(model.weight), decay_param_ids)
        self.assertIn(id(model.bias), no_decay_param_ids)
        self.assertIn(id(model.lora_alpha), decay_param_ids)
        self.assertIn(id(model.alpha), no_decay_param_ids)
        self.assertIn(id(model.beta), no_decay_param_ids)
        self.assertIn(id(model.gamma), no_decay_param_ids)
        self.assertIn(id(model._parameters["lambda"]), no_decay_param_ids)
        self.assertFalse(decay_param_ids & no_decay_param_ids)

    def test_train_entry_builds_mfnet_trainer_with_sgd_and_scheduler(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            config_path = config_dir / "external_config.jsonc"
            config_text = '{"train": {"max_epochs": 1}}\n'
            config_path.write_text(config_text, encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": "/tmp/pretrained.pth",
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=self._make_train_entry_config(root_dir=str(work_dir)),
                default_work_dir=work_dir,
            )

            dataset_calls = result["dataset_calls"]
            self.assertEqual(len(dataset_calls), 2)
            self.assertEqual(dataset_calls[0]["name"], "vaihingen")
            self.assertEqual(dataset_calls[1]["name"], "vaihingen")
            self.assertEqual(dataset_calls[0]["split"], "train")
            self.assertEqual(dataset_calls[1]["split"], "val")
            self.assertEqual(dataset_calls[0]["ids"], ["1"])
            self.assertEqual(dataset_calls[1]["ids"], ["5"])
            self.assertEqual(dataset_calls[0]["tile_sampling_weights"], [2.5])
            self.assertEqual(dataset_calls[0]["patch_sampling"], {"enabled": False})
            self.assertNotIn("tile_sampling_weights", dataset_calls[1])
            self.assertNotIn("patch_sampling", dataset_calls[1])
            expected_dsm_preprocessing = self._make_train_entry_config(root_dir=str(work_dir))["dataset"][
                "dsm_preprocessing"
            ]
            self.assertEqual(dataset_calls[0]["dsm_preprocessing"], expected_dsm_preprocessing)
            self.assertEqual(dataset_calls[1]["dsm_preprocessing"], expected_dsm_preprocessing)
            self.assertEqual(result["load_config_paths"], [str(config_path)])
            self.assertEqual(len(result["default_work_dir_calls"]), 1)
            self.assertEqual(
                result["default_work_dir_calls"][0],
                {
                    "model_name": "mfnet_unetformer",
                    "dataset_name": "vaihingen",
                    "lambda_align": None,
                    "root_dir": "work_dirs",
                },
            )
            self.assertEqual(len(result["trainer_kwargs"]), 1)
            self.assertEqual(result["trainer_classes"], ["MFNetTrainer"])
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["optimizer"], torch.optim.SGD)
            self.assertIsInstance(
                trainer_kwargs["scheduler"],
                torch.optim.lr_scheduler.MultiStepLR,
            )
            self.assertEqual(len(result["model_cfg"]), 1)
            self.assertEqual(result["model_cfg"][0]["sam_checkpoint"], "/tmp/sam_vit_b.pth")
            self.assertIsInstance(trainer_kwargs["logger"], _TestNetLogger)
            self.assertTrue(trainer_kwargs["logger"].use_tensorboard)
            self.assertEqual(trainer_kwargs["cfg"]["val_epoch_interval"], 1)
            self.assertEqual(trainer_kwargs["cfg"]["batch_size"], 2)
            self.assertEqual(trainer_kwargs["cfg"]["experiment_name"], work_dir.name)
            self.assertEqual(trainer_kwargs["cfg"]["sam_checkpoint"], "/tmp/sam_vit_b.pth")
            self.assertIsNone(trainer_kwargs["cfg"]["resume_from"])
            self.assertEqual(trainer_kwargs["cfg"]["load_from"], "/tmp/pretrained.pth")
            self.assertEqual(trainer_kwargs["cfg"]["work_dir"], str(work_dir))
            self.assertNotIn("checkpoint_manager", trainer_kwargs)
            self.assertEqual(
                trainer_kwargs["cfg"]["validation"],
                {"stride": 32},
            )
            self.assertNotIn("effective_batch_size", trainer_kwargs["cfg"])
            saved_cfg = json.loads((work_dir / config_path.name).read_text(encoding="utf-8"))
            self.assertEqual(saved_cfg["model"]["type"], "mfnet_unetformer")
            self.assertEqual(saved_cfg["model"]["sam_checkpoint"], "/tmp/sam_vit_b.pth")
            self.assertEqual(saved_cfg["dataset"]["name"], "vaihingen")
            self.assertEqual(config_path.read_text(encoding="utf-8"), config_text)
            self.assertFalse((work_dir / "train_config.jsonc").exists())

    def test_train_entry_requires_dsm_preprocessing_config(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            del cfg["dataset"]["dsm_preprocessing"]  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                    "model_type": None,
                },
            )()

            with self.assertRaisesRegex(KeyError, "dsm_preprocessing"):
                self._run_train_entry(
                    module=module,
                    args=args,
                    cfg=cfg,
                    default_work_dir=work_dir,
                )

    def test_train_entry_model_type_overrides_config_for_new_experiment(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            config_path = config_dir / "external_config.jsonc"
            config_text = '{"model": {"type": "mfnet_unetformer"}}\n'
            config_path.write_text(config_text, encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                    "model_type": "testnet_dga10",
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=self._make_train_entry_config(root_dir=str(work_dir)),
                default_work_dir=work_dir,
            )

            self.assertEqual(result["default_work_dir_calls"][0]["model_name"], "testnet_dga10")
            self.assertEqual(result["model_cfg"][0]["type"], "testnet_dga10")
            self.assertEqual(result["trainer_classes"], ["MFNetDGATrainer"])
            saved_cfg = json.loads((work_dir / config_path.name).read_text(encoding="utf-8"))
            self.assertEqual(saved_cfg["model"]["type"], "testnet_dga10")
            self.assertEqual(config_path.read_text(encoding="utf-8"), config_text)

    def test_train_entry_saves_merged_config_using_child_config_name(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            base_config_path = config_dir / "base_config.jsonc"
            child_config_path = config_dir / "child_config.jsonc"
            base_cfg = self._make_train_entry_config(root_dir=str(Path(tmpdir) / "dataset_root"))
            base_cfg["train"]["batch_size"] = 2  # type: ignore[index]
            base_cfg["dataset"]["train_ids"] = ["1", "2"]  # type: ignore[index]
            base_config_path.write_text(json.dumps(base_cfg, indent=4) + "\n", encoding="utf-8")
            child_config_text = """
            {
              "extends": "base_config.jsonc",
              "dataset": {
                "train_ids": ["3"]
              },
              "train": {
                "batch_size": 4
              }
            }
            """
            child_config_path.write_text(child_config_text, encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            args = type(
                "Args",
                (),
                {
                    "config": str(child_config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                    "model_type": "testnet_dga10",
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=None,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["load_config_paths"], [str(child_config_path)])
            self.assertFalse((work_dir / base_config_path.name).exists())
            saved_config_path = work_dir / child_config_path.name
            self.assertTrue(saved_config_path.is_file())
            saved_cfg = json.loads(saved_config_path.read_text(encoding="utf-8"))
            self.assertNotIn("extends", saved_cfg)
            self.assertEqual(saved_cfg["model"]["type"], "testnet_dga10")
            self.assertEqual(saved_cfg["model"]["num_classes"], 6)
            self.assertEqual(saved_cfg["dataset"]["train_ids"], ["3"])
            self.assertEqual(saved_cfg["dataset"]["val_ids"], ["5"])
            self.assertEqual(saved_cfg["train"]["batch_size"], 4)
            self.assertEqual(saved_cfg["train"]["max_epochs"], 1)
            self.assertEqual(child_config_path.read_text(encoding="utf-8"), child_config_text)

    def test_train_entry_uses_dga_trainer_and_logger_for_dga_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "testnet_dga10"  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetDGATrainer"])
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], _TestNetRecorderLogger)

    def test_train_entry_uses_dga_trainer_and_logger_for_dga2_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "testnet_dga20"  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetDGATrainer"])
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], _TestNetRecorderLogger)

    def test_train_entry_uses_dga_trainer_and_logger_for_dga20_dgsf10_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "testnet_dga20_dgsf10"  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetDGATrainer"])
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], _TestNetRecorderLogger)

    def test_train_entry_uses_base_trainer_and_recorder_logger_for_dgsf10_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "testnet_dgsf10"  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetTrainer"])
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], _TestNetRecorderLogger)

    def test_train_entry_uses_base_trainer_and_recorder_logger_for_dgfm_models(self) -> None:
        module = self._load_train_entry_module()
        for model_type in [
            "testnet_dgfm",
            "testnet_dgfm01",
            "testnet_dgfm01_upernet",
            "testnet_sgcf",
        ]:
            with self.subTest(model_type=model_type):
                with tempfile.TemporaryDirectory() as tmpdir:
                    config_path = Path(tmpdir) / "external_config.jsonc"
                    config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
                    work_dir = Path(tmpdir) / "auto_work"
                    cfg = self._make_train_entry_config(root_dir=str(work_dir))
                    cfg["model"]["type"] = model_type  # type: ignore[index]
                    args = type(
                        "Args",
                        (),
                        {
                            "config": str(config_path),
                            "device": "cpu",
                            "resume_dir": None,
                            "resume_ckpt": None,
                            "load_from": None,
                        },
                    )()

                    result = self._run_train_entry(
                        module=module,
                        args=args,
                        cfg=cfg,
                        default_work_dir=work_dir,
                    )

                    self.assertEqual(result["trainer_classes"], ["MFNetTrainer"])
                    trainer_kwargs = result["trainer_kwargs"][0]
                    self.assertIsInstance(trainer_kwargs["logger"], _TestNetRecorderLogger)

    def test_train_entry_uses_dga_trainer_and_logger_for_dga3_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "testnet_dga30"  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetDGATrainer"])
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], _TestNetRecorderLogger)

    def test_train_entry_uses_baseline_auxalign_trainer_for_baseline_auxalign_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "testnet_auxalign"  # type: ignore[index]
            cfg["train"]["lambda_align"] = 0.5  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetBaselineAuxAlignTrainer"])
            self.assertEqual(result["default_work_dir_calls"][0]["lambda_align"], 0.5)
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], _TestNetLogger)
            self.assertEqual(trainer_kwargs["cfg"]["lambda_align"], 0.5)

    def test_train_entry_uses_auxalign_trainer_for_auxalign_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "testnet_prealign_auxalign"  # type: ignore[index]
            cfg["train"]["lambda_align"] = 0.5  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetAuxAlignTrainer"])
            self.assertEqual(result["default_work_dir_calls"][0]["lambda_align"], 0.5)
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], _TestNetLogger)

    def test_train_entry_uses_auxalign_trainer_and_recorder_logger_for_prealign_auxalign_dgsf10(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "testnet_prealign_auxalign_dgsf10"  # type: ignore[index]
            cfg["train"]["lambda_align"] = 0.5  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetAuxAlignTrainer"])
            self.assertEqual(result["default_work_dir_calls"][0]["lambda_align"], 0.5)
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], _TestNetRecorderLogger)
            self.assertEqual(trainer_kwargs["cfg"]["lambda_align"], 0.5)

    def test_train_entry_uses_auxalign_dga_trainer_for_combined_model(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            cfg = self._make_train_entry_config(root_dir=str(work_dir))
            cfg["model"]["type"] = "testnet_prealign_auxalign_dga10"  # type: ignore[index]
            cfg["train"]["lambda_align"] = 0.5  # type: ignore[index]
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": None,
                    "load_from": None,
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=cfg,
                default_work_dir=work_dir,
            )

            self.assertEqual(result["trainer_classes"], ["MFNetAuxAlignDGATrainer"])
            self.assertEqual(result["default_work_dir_calls"][0]["lambda_align"], 0.5)
            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertIsInstance(trainer_kwargs["logger"], _TestNetRecorderLogger)
            self.assertEqual(trainer_kwargs["cfg"]["lambda_align"], 0.5)

    def test_train_entry_uses_resume_ckpt_for_new_experiment_only(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "external_config.jsonc"
            config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            work_dir = Path(tmpdir) / "auto_work"
            resume_ckpt = Path(tmpdir) / "manual_resume.pth"
            args = type(
                "Args",
                (),
                {
                    "config": str(config_path),
                    "device": "cpu",
                    "resume_dir": None,
                    "resume_ckpt": str(resume_ckpt),
                    "load_from": "/tmp/pretrained.pth",
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=self._make_train_entry_config(root_dir=str(work_dir)),
                default_work_dir=work_dir,
            )

            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertEqual(result["load_config_paths"], [str(config_path)])
            self.assertEqual(trainer_kwargs["cfg"]["work_dir"], str(work_dir))
            self.assertEqual(trainer_kwargs["cfg"]["resume_from"], str(resume_ckpt))
            self.assertEqual(trainer_kwargs["cfg"]["load_from"], "/tmp/pretrained.pth")
            self.assertTrue((work_dir / config_path.name).is_file())

    def test_train_entry_resume_dir_overrides_file_parameters(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_dir = Path(tmpdir) / "resume_work"
            resume_dir.mkdir()
            resume_config_path = resume_dir / "resume_config.json"
            resume_config_text = '{"train": {"max_epochs": 9}}\n'
            resume_config_path.write_text(resume_config_text, encoding="utf-8")
            existing_log_path = resume_dir / "train.log"
            existing_log_text = "Experiment: existing-run\n"
            existing_log_path.write_text(existing_log_text, encoding="utf-8")
            external_config_path = Path(tmpdir) / "external_config.jsonc"
            external_config_path.write_text('{"train": {"max_epochs": 1}}\n', encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "config": str(external_config_path),
                    "device": "cpu",
                    "resume_dir": str(resume_dir),
                    "resume_ckpt": str(Path(tmpdir) / "manual_resume.pth"),
                    "load_from": "/tmp/pretrained.pth",
                },
            )()

            result = self._run_train_entry(
                module=module,
                args=args,
                cfg=self._make_train_entry_config(root_dir=str(resume_dir)),
                default_work_dir=None,
            )

            trainer_kwargs = result["trainer_kwargs"][0]
            self.assertEqual(result["load_config_paths"], [str(resume_config_path)])
            self.assertEqual(result["default_work_dir_calls"], [])
            self.assertEqual(trainer_kwargs["cfg"]["work_dir"], str(resume_dir))
            self.assertEqual(trainer_kwargs["cfg"]["experiment_name"], resume_dir.name)
            self.assertEqual(trainer_kwargs["cfg"]["resume_from"], str(resume_dir / "latest.pth"))
            self.assertIsNone(trainer_kwargs["cfg"]["load_from"])
            self.assertEqual(resume_config_path.read_text(encoding="utf-8"), resume_config_text)
            self.assertEqual(existing_log_path.read_text(encoding="utf-8"), existing_log_text)
            self.assertFalse((resume_dir / external_config_path.name).exists())

    def test_train_entry_rejects_model_type_override_with_resume_dir(self) -> None:
        module = self._load_train_entry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            resume_dir = Path(tmpdir) / "resume_work"
            resume_dir.mkdir()
            resume_config_path = resume_dir / "resume_config.json"
            resume_config_text = '{"train": {"max_epochs": 9}}\n'
            resume_config_path.write_text(resume_config_text, encoding="utf-8")
            external_config_path = Path(tmpdir) / "external_config.jsonc"
            external_config_text = '{"train": {"max_epochs": 1}}\n'
            external_config_path.write_text(external_config_text, encoding="utf-8")
            args = type(
                "Args",
                (),
                {
                    "config": str(external_config_path),
                    "device": "cpu",
                    "resume_dir": str(resume_dir),
                    "resume_ckpt": None,
                    "load_from": None,
                    "model_type": "testnet_dga10",
                },
            )()

            with self.assertRaisesRegex(ValueError, "--model-type cannot be used with --resume-dir"):
                self._run_train_entry(
                    module=module,
                    args=args,
                    cfg=self._make_train_entry_config(root_dir=str(resume_dir)),
                    default_work_dir=None,
                )

            self.assertEqual(resume_config_path.read_text(encoding="utf-8"), resume_config_text)
            self.assertEqual(external_config_path.read_text(encoding="utf-8"), external_config_text)
            self.assertFalse((resume_dir / external_config_path.name).exists())
