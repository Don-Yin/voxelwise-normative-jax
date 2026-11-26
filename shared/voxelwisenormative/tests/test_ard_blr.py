"""Test the diagonal prior and a regression case where initialization matters."""

import numpy as np
import pytest

from shared.voxelwisenormative import ard_blr
from shared.voxelwisenormative.basis import Basis
from shared.voxelwisenormative.batched_blr import LOG2PI, PENALTY, fit_chunks
from shared.voxelwisenormative.cohort import standardise
from shared.voxelwisenormative.model import fit_score


def test_fixed_ard_matches_dense_gaussian_conditioning():
    """An independent subject-space calculation checks objective, warp and predictive variance."""
    x = np.linspace(-1, 1, 17)
    phi = np.column_stack((np.ones_like(x), x, 1 + x, x**2))
    y = np.column_stack((np.cos(3 * x), np.sin(2 * x)))
    hyp = np.array([[0.4, -0.13, 0.12, -0.3, 1.6, 0.4, 3.0], [-0.2, 0.3, 0.19, 2.1, -0.7, 4.0, 0.2]])
    p, pt = phi[:11], phi[11:]
    y_train, yt = y[:11], y[11:]
    _, evaluate = ard_blr.make_kernels(p.shape[1])
    objective, z = evaluate(hyp, *ard_blr.design(p, pt), y_train, yt)
    for i, (log_beta, loc, log_scale, *log_alpha) in enumerate(hyp):
        noise = np.exp(-log_beta)
        prior = np.exp(-np.array(log_alpha))
        u = np.exp(log_scale) * (np.arcsinh(y[:, i]) + loc)
        warped = np.sinh(u)
        log_jacobian = log_scale + np.log(np.cosh(u)) - 0.5 * np.log1p(y[:, i] ** 2)
        cov = (p * prior) @ p.T + noise * np.eye(len(p))
        cross = (pt * prior) @ p.T
        mean = cross @ np.linalg.solve(cov, warped[:len(p)])
        variance = noise + (pt**2) @ prior - np.einsum("ij,ji->i", cross, np.linalg.solve(cov, cross.T))
        expected_z = (warped[len(p):] - mean) / np.sqrt(variance)
        marginal = len(p) * LOG2PI + np.linalg.slogdet(cov)[1] + warped[:len(p)] @ np.linalg.solve(cov, warped[:len(p)])
        expected_loss = 0.5 * marginal - log_jacobian[:len(p)].sum() + PENALTY * np.linalg.norm(hyp[i])
        np.testing.assert_allclose(z[:, i], expected_z, rtol=0, atol=1e-10)
        np.testing.assert_allclose(objective[i], expected_loss, rtol=0, atol=1e-10)


def test_restarts_fix_a_worse_minimum_and_fit_score_preserves_chunks():
    """Oppositely skewed responses give a reproducible poor all-ones solution for one column."""
    rng = np.random.default_rng(7)
    size = 84
    held = np.arange(size) % 4 == 0
    batch = np.arange(size) % 3
    x = rng.normal(size=(size, 2))
    y = 0.2 * x[:, [0]] + 0.2 * (batch[:, None] == 0) + np.expm1(0.9 * rng.normal(size=(size, 2))) * [-1, 1]
    phi = Basis.fit(x[~held], batch[~held]).apply(x, batch)
    d = ard_blr.design(phi[~held], phi[held])
    train, test = standardise(y, held)
    single, _ = ard_blr.make_kernels(phi.shape[1])
    _, from_ones, _, _ = fit_chunks(single, d, train, test)
    hyp, best, z, gradient = fit_chunks(ard_blr.make_fitter(phi.shape[1]), d, train, test)
    assert np.all(best <= from_ones + 1e-9)
    assert from_ones[0] - best[0] > 1
    assert hyp.shape == gradient.shape == (y.shape[1], 3 + phi.shape[1])
    assert np.all(np.ptp(hyp[:, 3:], axis=1) > 1)
    np.testing.assert_allclose(fit_score(x, y, held, batch, ard=True, chunk=1), z, rtol=0, atol=1e-7)
    with pytest.raises(ValueError, match="not supported together"):
        fit_score(x, y, held, batch, ard=True, heteroskedastic=True)
