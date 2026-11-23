"""Experiment-local exact Newton variants; the shared solver remains the control."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from shared.voxelwisenormative.batched_blr import N_ITER

from .evidence import evidence_derivatives, full_evidence, objective, predict, projector

STEP_TOL = 1e-6
GRAD_TOL_PER_SUBJECT = 1e-7
STALL_DAMPING = 1e8


def make_variant(*, analytic: bool, early: bool, tf32: bool):
    """compile one solver with dtype determined by its inputs."""
    project = projector(tf32)
    loss = partial(objective, project=project)
    axes = (0, 1, None, None)
    f = jax.vmap(loss, in_axes=axes)
    if analytic:
        derivatives = jax.vmap(partial(evidence_derivatives, project=project), in_axes=axes)
    else:
        grad = jax.grad(loss)
        hess = jax.hessian(loss)

        def both(hyp, x, lam, G):
            """automatic derivatives of the same stabilized objective."""
            return grad(hyp, x, lam, G), hess(hyp, x, lam, G)

        derivatives = jax.vmap(both, in_axes=axes)
    zscore = jax.vmap(partial(predict, project=project), in_axes=(0, 1, 1, None, None, None), out_axes=1)

    @jax.jit
    def fit(lam, G, H, y, yt):
        """fit one voxel block and return hyperparameters, evidence, z and iteration counts."""
        x, xt = jnp.arcsinh(y), jnp.arcsinh(yt)
        voxels = y.shape[1]
        eye = jnp.eye(4, dtype=y.dtype)
        hyp = jnp.ones((voxels, 4), dtype=y.dtype)
        state = (hyp, jnp.full(voxels, 1e-3, dtype=y.dtype), f(hyp, x, lam, G),
                 jnp.ones(voxels, dtype=bool), jnp.zeros(voxels, dtype=jnp.int32))

        def step(_, carry):
            """one damped Newton proposal with a decrease test per voxel."""
            hyp, mu, fv, active, counts = carry
            grad, hess = derivatives(hyp, x, lam, G)
            delta = -jnp.linalg.solve(hess + mu[:, None, None] * eye, grad[..., None])[..., 0]
            delta = jnp.clip(delta, -2, 2)
            candidate = hyp + delta
            trial = f(candidate, x, lam, G)
            accept = active & jnp.isfinite(trial) & (trial < fv)
            hyp = jnp.where(accept[:, None], candidate, hyp)
            mu = jnp.where(accept, jnp.maximum(mu / 3, 1e-9), jnp.minimum(mu * 4, 1e30))
            if early:
                # Require both a small gradient and a small proposal, or numerical stalling.
                small_grad = jnp.max(jnp.abs(grad), axis=1) <= x.shape[0] * GRAD_TOL_PER_SUBJECT
                small_step = jnp.max(jnp.abs(delta), axis=1) <= STEP_TOL
                finish = ((small_grad & small_step) | (~accept & (mu >= STALL_DAMPING)))
                return hyp, mu, jnp.where(accept, trial, fv), active & ~finish, counts + active
            return hyp, mu, jnp.where(accept, trial, fv), active, counts

        if early:
            def keep_going(pair):
                """stop the block once all its voxels have finished."""
                index, carry = pair
                return (index < N_ITER) & jnp.any(carry[3])

            def next_step(pair):
                """advance the block and its loop counter."""
                index, carry = pair
                return index + 1, step(index, carry)

            _, (hyp, _, fv, _, counts) = jax.lax.while_loop(keep_going, next_step, (0, state))
        else:
            hyp, _, fv, _, counts = jax.lax.fori_loop(0, N_ITER, step, state)
            counts = counts + N_ITER
        return hyp, full_evidence(fv, y), zscore(hyp, x, xt, lam, G, H), counts

    return fit
