"""Batched Cholesky evidence, prediction and exact-gradient BFGS for warped BLR."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from .batched_blr import LOG2PI, PENALTY

GRAD_TOL = 3e-5
LINE_SEARCH_LIMIT = 24


class _State(NamedTuple):
    """Independent parameters, gradients and inverse Hessians in a fixed-shape batch."""

    h: jax.Array
    f: jax.Array
    g: jax.Array
    inv: jax.Array
    failures: jax.Array


def _inner(x, y):
    """Row-wise inner product."""
    return jnp.sum(x * y, axis=1)


def _outer(x, y):
    """Row-wise outer product."""
    return x[:, :, None] * y[:, None, :]


def _search(value_and_grad, state: _State, direction: jax.Array, active: jax.Array):
    """Backtrack each active voxel until its proposal passes the Armijo decrease test."""
    h, f, g, *_ = state
    slope = _inner(g, direction)

    def pending(ls):
        k, _, accepted, _, _, _ = ls
        return (k < LINE_SEARCH_LIMIT) & jnp.any(active & ~accepted)

    def trial(ls):
        k, step, accepted, best_h, best_f, best_g = ls
        todo = active & ~accepted
        proposal = jnp.where(todo[:, None], h + step * direction, best_h)
        ft, gt = value_and_grad(proposal)
        accept = todo & jnp.isfinite(ft) & jnp.isfinite(gt).all(axis=1) & (ft <= f + 1e-4 * step * slope)
        return (
            k + 1, step * 0.5, accepted | accept,
            jnp.where(accept[:, None], proposal, best_h),
            jnp.where(accept, ft, best_f),
            jnp.where(accept[:, None], gt, best_g),
        )

    _, _, accepted, ht, ft, gt = jax.lax.while_loop(pending, trial, (0, 1., jnp.zeros_like(active), h, f, g))
    return accepted, ht, ft, gt


def _minimize(value_and_grad, h0: jax.Array, max_iters: int):
    """Stop each voxel at gradient tolerance, two failed searches, or the iteration cap."""
    eye = jnp.eye(h0.shape[1], dtype=h0.dtype)
    f0, g0 = value_and_grad(h0)
    initial = _State(h0, f0, g0, jnp.broadcast_to(eye, (h0.shape[0], *eye.shape)), jnp.zeros(h0.shape[0], jnp.int32))

    def active(state: _State):
        return (jnp.max(jnp.abs(state.g), axis=1) > GRAD_TOL) & (state.failures < 2) & jnp.isfinite(state.f)

    def condition(carry: tuple[int, _State]):
        k, state = carry
        return (k < max_iters) & jnp.any(active(state))

    def step(carry: tuple[int, _State]):
        k, state = carry
        h, _, g, inv, failures = state
        running = active(state)
        d = -(inv @ g[:, :, None])[:, :, 0]
        d = jnp.where((_inner(g, d) < 0)[:, None], d, -g)
        d /= jnp.maximum(1, jnp.max(jnp.abs(d), axis=1, keepdims=True) / 2)
        passed, ht, ft, gt = _search(value_and_grad, state, d, running)
        s, delta = ht - h, gt - g
        sy = _inner(s, delta)
        update = passed & (sy > 1e-10 * jnp.sqrt(_inner(s, s) * _inner(delta, delta)))
        inv_sy = 1 / jnp.where(update, sy, 1)
        hdy = (inv @ delta[:, :, None])[:, :, 0]
        scale = (sy + _inner(delta, hdy)) * inv_sy**2
        next_inv = inv + scale[:, None, None] * _outer(s, s)
        next_inv -= inv_sy[:, None, None] * (_outer(hdy, s) + _outer(s, hdy))
        inv = jnp.where(update[:, None, None], next_inv, inv)
        inv = jnp.where((running & ~passed)[:, None, None], eye, inv)
        failures = jnp.where(passed, 0, failures + (running & ~passed))
        return k + 1, _State(ht, ft, gt, inv, failures)

    return jax.lax.while_loop(condition, step, (0, initial))[1]


def make_kernels(n_beta: int, n_alpha: int, noise_weights, *, max_iters: int = 300):
    """Build jit fit/evaluate kernels with a supplied log-precision and weighted Gram function.

    Hyperparameters are ordered [noise weights, warp location, log warp scale, log prior precision(s)].
    noise_weights(w, phi, noise, contraction) returns voxel-major log beta and weighted Gram.
    fit returns (hyperparameters, objective, test z, gradient); evaluate returns (objective, test z).
    """
    jax.config.update("jax_enable_x64", True)

    def response_transform(y):
        """Cache the response-only warp input and Jacobian term before optimization."""
        return jnp.arcsinh(y.T), 0.5 * jnp.log1p(y.T**2).sum(axis=1)

    def warp(h, y_arc):
        """Evaluate the sinh-arcsinh warp from voxel-major asinh observations."""
        u = jnp.exp(h[:, n_beta + 1:n_beta + 2]) * (y_arc + h[:, n_beta:n_beta + 1])
        return jnp.sinh(u), u

    def posterior(h, phi, noise, contraction, y_arc, jac_constant):
        """Factor each posterior precision and evaluate the penalized warped evidence."""
        n, d = phi.shape
        log_beta, gram = noise_weights(h[:, :n_beta], phi, noise, contraction)
        log_alpha = jnp.broadcast_to(h[:, n_beta + 2:], (h.shape[0], d))
        z, u = warp(h, y_arc)
        beta_z = jnp.exp(log_beta) * z
        chol = jnp.linalg.cholesky(gram + jnp.eye(d, dtype=phi.dtype) * jnp.exp(log_alpha)[:, None, :])
        white_rhs = solve_triangular(chol, (beta_z @ phi)[..., None], lower=True)[..., 0]
        logdet = 2 * jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=1)
        quad = (beta_z * z).sum(axis=1) - (white_rhs**2).sum(axis=1)
        logcosh = jnp.logaddexp(u, -u) - jnp.log(2.)
        logjac = n * h[:, n_beta + 1] + logcosh.sum(axis=1) - jac_constant
        noise_logdet = log_beta.sum(axis=1) * (n if log_beta.shape[1] == 1 else 1)
        marginal = 0.5 * (n * LOG2PI - noise_logdet - log_alpha.sum(axis=1) + logdet + quad)
        penalty = PENALTY * jnp.linalg.norm(h, axis=1)
        return marginal - logjac + penalty, chol, white_rhs

    def prediction(h, phit, noiset, yt, chol, white_rhs):
        """Divide held-out warped residuals by noise and posterior predictive uncertainty."""
        log_beta = h[:, :n_beta] @ noiset.T
        test_basis = jnp.broadcast_to(phit.T, (h.shape[0], *phit.T.shape))
        white_test = solve_triangular(chol, test_basis, lower=True)
        mean = (white_test * white_rhs[:, :, None]).sum(axis=1)
        variance = jnp.exp(-log_beta) + (white_test**2).sum(axis=1)
        zt, _ = warp(h, jnp.arcsinh(yt.T))
        return ((zt - mean) / jnp.sqrt(variance)).T

    @jax.jit
    def evaluate(hyp, phi, phit, noise, noiset, contraction, y, yt):
        """Return objective and held-out z without optimizing the supplied parameters."""
        y_arc, jac_constant = response_transform(y)
        f, chol, white_rhs = posterior(hyp, phi, noise, contraction, y_arc, jac_constant)
        return f, prediction(hyp, phit, noiset, yt, chol, white_rhs)

    @jax.jit
    def fit(phi, phit, noise, noiset, contraction, y, yt, initial=None):
        """Optimize a chunk from ones or a supplied (voxels, hyperparameters) matrix."""
        y_arc, jac_constant = response_transform(y)
        shape = (y.shape[1], n_beta + 2 + n_alpha)
        h0 = jnp.ones(shape, dtype=phi.dtype) if initial is None else jnp.asarray(initial, dtype=phi.dtype)
        if h0.shape != shape:
            raise ValueError(f"initial hyperparameters must have shape {shape}, got {h0.shape}")

        def total(h):
            f, _, _ = posterior(h, phi, noise, contraction, y_arc, jac_constant)
            return f.sum(), f

        def value_and_grad(h):
            (_, f), g = jax.value_and_grad(total, has_aux=True)(h)
            return f, g

        solved = _minimize(value_and_grad, h0, max_iters)
        _, chol, white_rhs = posterior(solved.h, phi, noise, contraction, y_arc, jac_constant)
        z = prediction(solved.h, phit, noiset, yt, chol, white_rhs)
        return solved.h, solved.f, z, solved.g

    return fit, evaluate
