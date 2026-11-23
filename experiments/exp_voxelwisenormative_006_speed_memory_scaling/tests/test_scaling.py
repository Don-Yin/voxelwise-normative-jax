"""CPU checks on synthetic data only: strided columns and one measured case per A100 variant."""

from __future__ import annotations

import unittest

import numpy as np
from src.modules.gpu_scaling import Inputs, measure, strided_columns

from shared.voxelwisenormative.batched_blr import design


class ScalingTests(unittest.TestCase):
    """no IXI data: random designs and responses with the benchmark's subject counts."""

    def test_strided_columns(self):
        """every mask column at full size, unique and in mask order otherwise."""
        np.testing.assert_array_equal(strided_columns(165402, 165402), np.arange(165402))
        for n in (1000, 32768, 100000):
            columns = strided_columns(n, 165402)
            self.assertEqual(columns.size, n)
            self.assertTrue(np.all(np.diff(columns) > 0))
            self.assertEqual((columns[0], columns[-1]), (0, 165401))

    def test_measure_both_variants(self):
        """both variants run, report timings and finite z, with block widths covering every voxel."""
        import jax

        jax.config.update("jax_enable_x64", True)
        rng = np.random.default_rng(0)
        phi = np.column_stack([np.ones(563), rng.normal(size=(563, 11))])
        test = np.zeros(563, bool)
        test[rng.choice(563, 113, replace=False)] = True
        y = rng.normal(size=(563, 70))
        inputs = Inputs(y[~test], y[test], design(phi[~test], phi[test]), np.arange(70),
                        np.array([], dtype=np.int64), np.zeros((113, 0)))
        for variant in ("early-f32", "baseline"):
            result = measure(variant, inputs, chunk=32, repeats=2, device=jax.devices()[0])
            self.assertEqual(result["block_widths"], [32, 32, 6])
            self.assertEqual(len(result["seconds"]), 2)
            self.assertEqual(result["nonfinite_z"], 0)
            self.assertTrue(result["gate_passed"])
            self.assertIsNone(result["z_absdiff_max"])


if __name__ == "__main__":
    unittest.main()
