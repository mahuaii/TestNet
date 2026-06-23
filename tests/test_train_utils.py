from __future__ import annotations

import unittest

import torch

from utils import (
    build_default_work_dir,
    build_optimizer_param_groups,
    safe_path_component,
    work_dir_model_suffix,
)


class GateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.norm = torch.nn.BatchNorm1d(2)
        self.Adapter = torch.nn.Linear(2, 2)
        self.frozen = torch.nn.Linear(2, 2)
        self.frozen.requires_grad_(False)
        self.alpha = torch.nn.Parameter(torch.ones(1))
        self.block = torch.nn.Module()
        self.block.beta = torch.nn.Parameter(torch.ones(1))
        self.block.gamma = torch.nn.Parameter(torch.ones(1))
        self.block.lambda_param = torch.nn.Parameter(torch.ones(1))
        self.block.lambda_ = torch.nn.Parameter(torch.ones(1))
        self.block.register_parameter("lambda", torch.nn.Parameter(torch.ones(1)))


class TrainUtilsTest(unittest.TestCase):
    def test_safe_path_component_sanitizes_or_uses_fallback(self) -> None:
        self.assertEqual(safe_path_component(" Vaihingen Dataset! ", "dataset"), "vaihingen_dataset")
        self.assertEqual(safe_path_component("!!!", "dataset"), "dataset")

    def test_work_dir_model_suffix_removes_common_mfnet_tokens(self) -> None:
        self.assertEqual(work_dir_model_suffix("testnet_prealign_dga10"), "prealign_dga10")
        self.assertEqual(
            work_dir_model_suffix("testnet_prealign_auxalign_dga10"),
            "prealign_auxalign_dga10",
        )
        self.assertEqual(
            work_dir_model_suffix("testnet_prealign_mmadapter10"),
            "prealign_mmadapter10",
        )
        self.assertEqual(work_dir_model_suffix("testnet_prealign_spmf20"), "prealign_spmf20")
        self.assertEqual(work_dir_model_suffix("mfnet_unetformer"), "base")

    def test_build_default_work_dir_uses_short_run_id_instead_of_timestamp(self) -> None:
        path = build_default_work_dir(
            model_name="testnet_prealign_auxalign",
            dataset_name="Vaihingen Dataset",
            lambda_align=0.5,
            root_dir="runs",
        )

        self.assertEqual(path.parent.name, "runs")
        parts = path.name.split("_")
        self.assertEqual(parts[0:4], ["vaihingen", "dataset", "prealign", "auxalign"])
        self.assertEqual(parts[-2], "lambda-0.5")
        self.assertRegex(path.name, r"_lambda-0.5_[0-9a-f]{5}$")

    def test_build_optimizer_param_groups_applies_lr_and_weight_decay_rules(self) -> None:
        model = GateModel()

        param_groups = build_optimizer_param_groups(
            model,
            weight_decay=0.123,
            base_lr=0.01,
            adapter_lr=0.001,
        )

        self.assertEqual(len(param_groups), 4)
        regular_decay = next(
            group for group in param_groups if group["lr"] == 0.01 and group["weight_decay"] == 0.123
        )
        regular_no_decay = next(
            group for group in param_groups if group["lr"] == 0.01 and group["weight_decay"] == 0.0
        )
        adapter_decay = next(
            group for group in param_groups if group["lr"] == 0.001 and group["weight_decay"] == 0.123
        )
        adapter_no_decay = next(
            group for group in param_groups if group["lr"] == 0.001 and group["weight_decay"] == 0.0
        )
        regular_decay_ids = {id(param) for param in regular_decay["params"]}
        regular_no_decay_ids = {id(param) for param in regular_no_decay["params"]}
        adapter_decay_ids = {id(param) for param in adapter_decay["params"]}
        adapter_no_decay_ids = {id(param) for param in adapter_no_decay["params"]}
        named_params = dict(model.named_parameters())

        self.assertIn(id(named_params["backbone.weight"]), regular_decay_ids)
        self.assertIn(id(named_params["backbone.bias"]), regular_no_decay_ids)
        self.assertIn(id(named_params["norm.weight"]), regular_no_decay_ids)
        self.assertIn(id(named_params["norm.bias"]), regular_no_decay_ids)
        self.assertIn(id(named_params["Adapter.weight"]), adapter_decay_ids)
        self.assertIn(id(named_params["Adapter.bias"]), adapter_no_decay_ids)
        self.assertIn(id(named_params["alpha"]), regular_no_decay_ids)
        self.assertIn(id(named_params["block.beta"]), regular_no_decay_ids)
        self.assertIn(id(named_params["block.gamma"]), regular_no_decay_ids)
        self.assertIn(id(named_params["block.lambda"]), regular_no_decay_ids)
        self.assertIn(id(named_params["block.lambda_param"]), regular_no_decay_ids)
        self.assertIn(id(named_params["block.lambda_"]), regular_no_decay_ids)
        grouped_ids = {
            id(param)
            for group in param_groups
            for param in group["params"]
        }
        self.assertNotIn(id(named_params["frozen.weight"]), grouped_ids)
        self.assertNotIn(id(named_params["frozen.bias"]), grouped_ids)

    def test_build_optimizer_param_groups_rejects_invalid_lr_values(self) -> None:
        model = GateModel()

        with self.assertRaisesRegex(ValueError, "base_lr must be positive"):
            build_optimizer_param_groups(model, weight_decay=0.1, base_lr=0.0)
        with self.assertRaisesRegex(ValueError, "adapter_lr must be positive"):
            build_optimizer_param_groups(model, weight_decay=0.1, adapter_lr=0.0)


if __name__ == "__main__":
    unittest.main()
