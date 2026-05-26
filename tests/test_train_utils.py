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
        self.assertEqual(work_dir_model_suffix("mfnet_unetformer_prealign_dga10"), "prealign_dga10")
        self.assertEqual(
            work_dir_model_suffix("mfnet_unetformer_prealign_auxalign_dga10"),
            "prealign_auxalign_dga10",
        )
        self.assertEqual(work_dir_model_suffix("mfnet_unetformer"), "base")

    def test_build_default_work_dir_uses_short_run_id_instead_of_timestamp(self) -> None:
        path = build_default_work_dir(
            model_name="mfnet_unetformer_prealign_auxalign",
            dataset_name="Vaihingen Dataset",
            seed=80,
            lambda_align=0.5,
            root_dir="runs",
        )

        self.assertEqual(path.parent.name, "runs")
        parts = path.name.split("_")
        self.assertEqual(parts[0:5], ["vaihingen", "dataset", "prealign", "auxalign", "80"])
        self.assertEqual(parts[-2], "lambda-0.5")
        self.assertRegex(path.name, r"_lambda-0.5_[0-9a-f]{5}$")

    def test_build_optimizer_param_groups_exempts_gate_parameter_names(self) -> None:
        model = GateModel()

        param_groups = build_optimizer_param_groups(model, weight_decay=0.123)

        self.assertEqual(len(param_groups), 2)
        decay_group = next(group for group in param_groups if group["weight_decay"] == 0.123)
        no_decay_group = next(group for group in param_groups if group["weight_decay"] == 0.0)
        decay_param_ids = {id(param) for param in decay_group["params"]}
        no_decay_param_ids = {id(param) for param in no_decay_group["params"]}
        named_params = dict(model.named_parameters())

        self.assertIn(id(named_params["backbone.weight"]), decay_param_ids)
        self.assertIn(id(named_params["backbone.bias"]), decay_param_ids)
        self.assertIn(id(named_params["alpha"]), no_decay_param_ids)
        self.assertIn(id(named_params["block.beta"]), no_decay_param_ids)
        self.assertIn(id(named_params["block.gamma"]), no_decay_param_ids)
        self.assertIn(id(named_params["block.lambda"]), no_decay_param_ids)
        self.assertIn(id(named_params["block.lambda_param"]), decay_param_ids)
        self.assertIn(id(named_params["block.lambda_"]), decay_param_ids)


if __name__ == "__main__":
    unittest.main()
