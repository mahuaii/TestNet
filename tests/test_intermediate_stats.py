from __future__ import annotations

import unittest

import torch

from utils import IntermediateStatsRecorder


class IntermediateStatsRecorderTest(unittest.TestCase):
    def test_record_scalar_detaches_and_exports_float(self) -> None:
        recorder = IntermediateStatsRecorder(prefix="dga")
        value = torch.tensor(2.5, requires_grad=True)

        recorder.record_scalar("alpha", value)

        self.assertEqual(recorder.snapshot(), {"dga/alpha": 2.5})

    def test_record_scalar_rejects_non_scalar_tensors(self) -> None:
        recorder = IntermediateStatsRecorder()

        with self.assertRaisesRegex(ValueError, "expected a scalar tensor"):
            recorder.record_scalar("bad", torch.ones(2))

    def test_record_mean_std_uses_population_std(self) -> None:
        recorder = IntermediateStatsRecorder()

        recorder.record_mean_std("gate_fuse/g1", torch.tensor([1.0, 2.0, 3.0]))

        stats = recorder.snapshot()
        self.assertAlmostEqual(stats["gate_fuse/g1_mean"], 2.0)
        self.assertAlmostEqual(stats["gate_fuse/g1_std"], (2.0 / 3.0) ** 0.5)

    def test_record_norm_and_ratio_do_not_keep_graph(self) -> None:
        recorder = IntermediateStatsRecorder()
        numerator = torch.tensor([3.0, 4.0], requires_grad=True)
        denominator = torch.tensor([0.0, 10.0], requires_grad=True)

        recorder.record_norm("residual/y1_strength", numerator)
        recorder.record_norm_ratio("feature_ratio/y1_over_p1", numerator, denominator)

        stats = recorder.snapshot()
        self.assertAlmostEqual(stats["residual/y1_strength"], 5.0)
        self.assertAlmostEqual(stats["feature_ratio/y1_over_p1"], 0.5)

    def test_snapshot_can_reset(self) -> None:
        recorder = IntermediateStatsRecorder()
        recorder.record_scalar("depth_aggregation/w1", 0.25)

        self.assertEqual(recorder.snapshot(reset=True), {"depth_aggregation/w1": 0.25})
        self.assertEqual(recorder.snapshot(), {})

    def test_disabled_recorder_ignores_records(self) -> None:
        recorder = IntermediateStatsRecorder()

        with recorder.disabled():
            recorder.record_scalar("ignored", 1.0)
        recorder.record_scalar("kept", 2.0)

        self.assertEqual(recorder.snapshot(), {"kept": 2.0})


if __name__ == "__main__":
    unittest.main()
