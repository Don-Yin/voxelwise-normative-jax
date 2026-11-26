"""Check heteroskedastic evidence, noise prediction and the package fit option."""
import numpy as np
import pytest

from shared.voxelwisenormative.batched_blr import LOG2PI, PENALTY, fit_chunks
from shared.voxelwisenormative.heteroskedastic_blr import design, make_kernels
from shared.voxelwisenormative.model import fit_score


def _data():
    x = np.linspace(-1, 1, 45)
    phi = np.column_stack((np.ones_like(x), x, x**2))
    noise = np.column_stack((x, np.ones_like(x)))
    y = x[:, None] * np.array([0.4, -0.2, 0.1]) + np.exp(x[:, None]) * np.random.default_rng(8).normal(size=(45, 3))
    held = np.arange(len(x)) % 4 == 0
    h = np.array([[-0.8, 0.1, 0.2, -0.3, 0.6], [0.3, 0.2, -0.1, 0.1, -0.4], [-0.7, -0.1, 0.3, 0.2, 0.5]])
    return x, y, held, phi, noise, h


def test_weighted_gram_objective_and_held_out_variance_match_direct_solves():
    _, y, held, phi, noise, h = _data()
    train, test = y[~held], y[held]
    p, pt = phi[~held], phi[held]
    b, bt = noise[~held], noise[held]
    _, evaluate = make_kernels(b.shape[1])
    objective, scores = evaluate(h, *design(p, pt, b, bt), train, test)

    for i, (w0, w1, eps, log_scale, log_alpha) in enumerate(h):
        log_beta, log_beta_test = b @ [w0, w1], bt @ [w0, w1]
        beta, alpha = np.exp(log_beta), np.exp(log_alpha)
        scale = np.exp(log_scale)
        u = scale * (np.arcsinh(train[:, i]) + eps)
        z = np.sinh(u)
        precision = p.T @ (beta[:, None] * p) + alpha * np.eye(p.shape[1])
        m = np.linalg.solve(precision, p.T @ (beta * z))
        quad = beta @ (z - p @ m)**2 + alpha * (m @ m)
        f = 0.5 * (len(p) * LOG2PI - log_beta.sum() - p.shape[1] * log_alpha
                   + np.linalg.slogdet(precision).logabsdet + quad)
        jac = np.log(scale * np.cosh(u) / np.sqrt(1 + train[:, i]**2)).sum()
        expected_objective = f - jac + PENALTY * np.linalg.norm(h[i])
        variance = np.exp(-log_beta_test) + (pt * np.linalg.solve(precision, pt.T).T).sum(axis=1)
        zt = np.sinh(scale * (np.arcsinh(test[:, i]) + eps))
        np.testing.assert_allclose(objective[i], expected_objective, rtol=0, atol=1e-11)
        np.testing.assert_allclose(scores[:, i], (zt - pt @ m) / np.sqrt(variance), rtol=0, atol=1e-11)


def test_chunked_custom_starts_and_final_autodiff_gradient():
    _, y, held, phi, noise, start = _data()
    train, test = y[~held], y[held]
    d = design(phi[~held], phi[held], noise[~held], noise[held])
    fit, evaluate = make_kernels(noise.shape[1], max_iters=1)
    h, f, z, grad = fit_chunks(fit, d, train, test, chunk=2, initial=start)
    original, _ = evaluate(start, *d, train, test)
    expected_f, expected_z = evaluate(h, *d, train, test)
    assert np.all(f < original)
    np.testing.assert_allclose(f, expected_f, atol=1e-11, rtol=0)
    np.testing.assert_allclose(z, expected_z, atol=1e-11, rtol=0)
    for i, shift in enumerate(1e-5 * np.eye(h.shape[1])):
        higher, _ = evaluate(h + shift, *d, train, test)
        lower, _ = evaluate(h - shift, *d, train, test)
        np.testing.assert_allclose(grad[:, i], (higher - lower) / 2e-5, atol=1e-6, rtol=1e-7)


def test_public_fit_score_matches_explicit_training_scaled_noise_design():
    x, y, held, _, noise, _ = _data()
    noise[:, 0] = (x - x[~held].mean()) / x[~held].std()
    from_default = fit_score(x[:, None], y, held, heteroskedastic=True, chunk=2)
    from_explicit = fit_score(x[:, None], y, held, heteroskedastic=True, variance_design=noise, chunk=3)
    assert from_default.shape == y[held].shape and np.isfinite(from_default).all()
    np.testing.assert_allclose(from_default, from_explicit, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="include a column of ones"):
        fit_score(x[:, None], y, held, heteroskedastic=True, variance_design=x[:, None])
    with pytest.raises(ValueError, match="variance_design requires heteroskedastic"):
        fit_score(x[:, None], y, held, variance_design=noise)
