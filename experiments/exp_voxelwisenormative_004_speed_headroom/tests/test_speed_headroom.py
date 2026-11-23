"""Fast CPU checks for the analytic Newton step and full-mask accuracy gate."""

from __future__ import annotations

import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from src.modules.dataset import Dataset, agreement
from src.modules.dispatch import shards
from src.modules.evidence import evidence_derivatives, objective, projector

from shared.voxelwisenormative.batched_blr import design


class SpeedHeadroomTests(unittest.TestCase):
    """exercise the algebra and sharding without a CUDA device."""

    @classmethod
    def setUpClass(cls):
        """request float64 so analytic and automatic derivatives can be compared tightly."""
        jax.config.update("jax_enable_x64", True)

    def test_derivatives(self):
        """gradient and Hessian equal autodiff across several nonsingular parameter vectors."""
        rng = np.random.default_rng(42)
        phi = rng.normal(size=(80, 7))
        phi[:, -1] = phi[:, 1]
        d = design(phi, phi[:20])
        x, lam, G = (jnp.asarray(a) for a in (np.arcsinh(rng.normal(size=80)), d.lam, d.G))
        project = projector(False)
        loss = partial(objective, x=x, lam=lam, G=G, project=project)
        for h in ([1, 1, 1, 1], [-0.3, 0.05, -0.1, 2.2], [0.1, -0.2, 0.05, -2.0]):
            hyp = jnp.array(h, dtype=jnp.float64)
            grad, hess = evidence_derivatives(hyp, x, lam, G, project)
            np.testing.assert_allclose(grad, jax.grad(loss)(hyp), atol=1e-8, rtol=1e-10)
            np.testing.assert_allclose(hess, jax.hessian(loss)(hyp), atol=1e-8, rtol=1e-10)

    def test_gate(self):
        """reject NaN outside the reference voxels and errors above the reference threshold."""
        d = design(np.ones((3, 1)), np.ones((2, 1)))
        y, yt = np.zeros((3, 4)), np.zeros((2, 4))
        data = Dataset(y, yt, d, np.array([0, 2]), np.zeros((2, 2)), np.zeros(2))
        z = yt.copy()
        fv = np.zeros(4)
        z[0, 2] = 0.01
        self.assertTrue(agreement(data, fv, z)["passed"])
        z[0, 2] = 0.010001
        self.assertFalse(agreement(data, fv, z)["passed"])
        z[0, 2], z[1, 3] = 0, np.nan
        invalid = agreement(data, fv, z)
        self.assertEqual(invalid["nonfinite_z"], 1)
        self.assertEqual(invalid["z_absdiff_max"], 0)
        self.assertFalse(invalid["passed"])

    def test_shards(self):
        """cover the mask in order and balance the remainder across devices."""
        devices = tuple(range(4))
        ranges = shards(165402, 32768, devices)
        reconstructed = np.concatenate([np.arange(cols.start, cols.stop) for _, cols in ranges])
        np.testing.assert_array_equal(reconstructed, np.arange(165402))
        loads = [sum(cols.stop - cols.start for d, cols in ranges if d == device) for device in devices]
        self.assertEqual(max(loads) - min(loads), 1)
        self.assertTrue(all(cols.stop - cols.start <= 32768 for _, cols in ranges))
