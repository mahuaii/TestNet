from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from train import set_reproducibility


class ReproducibilityTest(unittest.TestCase):
    def test_set_reproducibility_resets_random_sequences(self) -> None:
        set_reproducibility(42)
        python_values = [random.random() for _ in range(3)]
        numpy_values = np.random.rand(3)
        torch_values = torch.rand(3)

        set_reproducibility(42)
        self.assertEqual(python_values, [random.random() for _ in range(3)])
        np.testing.assert_allclose(numpy_values, np.random.rand(3))
        torch.testing.assert_close(torch_values, torch.rand(3))


if __name__ == "__main__":
    unittest.main()
