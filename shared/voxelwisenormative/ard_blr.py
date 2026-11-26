"""Warped Bayesian linear regression with one prior precision per mean-basis weight."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .batched_blr import fit_chunks
from .batched_solve import make_kernels as _make_kernels

MAX_ITERS = 300
# Noise log precision, warp location, log warp scale, and the common initial log prior precision.
# Keep the reference start; cover identity warp and both sides of the warp-location tradeoff.
STARTS = (
    (1., 1., 1., 1.),
    (0., 0., 0., 1.),
    (0., 0., 0., 5.),
    (0., 0., 0., 9.),
    (1., -1., 1., 1.),
    (2., 0.5, -0.5, 5.),
    (2., -0.5, -0.5, 5.),
)


def design(phi: np.ndarray, phit: np.ndarray) -> tuple[np.ndarray, ...]:
    """Prepare the original mean designs, intercept-only noise designs and shared Gram matrix."""
    phi, phit = (np.asarray(a, dtype=np.float64) for a in (phi, phit))
    if phi.ndim != 2 or phit.ndim != 2 or not all(phi.shape) or not all(phit.shape) or phi.shape[1] != phit.shape[1]:
        raise ValueError(f"expected nonempty train/test mean designs with the same columns, got {phi.shape}, {phit.shape}")
    if not (np.isfinite(phi).all() and np.isfinite(phit).all()):
        raise ValueError("nonfinite values in the ARD mean design")
    return phi, phit, np.ones((len(phi), 1)), np.ones((len(phit), 1)), phi.T @ phi


def _noise_weights(w, _phi, _noise, gram):
    """Rescale the shared Gram matrix by the single noise precision per voxel."""
    return w, jnp.exp(w[:, :, None]) * gram


def make_kernels(n_weights: int, *, max_iters: int = MAX_ITERS):
    """Return the single-start fit and fixed-parameter evaluation in the original basis."""
    if n_weights < 1 or max_iters < 1:
        raise ValueError(f"ARD needs positive weights and iterations, got {n_weights=}, {max_iters=}")
    return _make_kernels(1, n_weights, _noise_weights, max_iters=max_iters)


def make_fitter(n_weights: int, *, max_iters: int = MAX_ITERS):
    """Fit independently from deterministic starts and keep the lowest objective for each voxel."""
    fit, _ = make_kernels(n_weights, max_iters=max_iters)

    @jax.jit
    def fit_restarts(phi, phit, noise, noiset, gram, y, yt):
        """Reuse one compiled optimizer in a serial scan so restarts need only one batch of memory."""
        n_voxels = y.shape[1]
        shape = (n_voxels, n_weights + 3)
        missing = jnp.full(shape, jnp.nan, dtype=phi.dtype)
        state = (missing, jnp.full(n_voxels, jnp.inf, dtype=phi.dtype),
                 jnp.full(yt.shape, jnp.nan, dtype=phi.dtype), missing)

        def take_best(best, start):
            """Keep the finite candidate with the better evidence for each response column."""
            seed = jnp.concatenate((start[:3], jnp.full(n_weights, start[3], dtype=phi.dtype)))
            h, f, z, g = fit(phi, phit, noise, noiset, gram, y, yt, jnp.broadcast_to(seed, shape))
            better = (jnp.isfinite(f) & (f < best[1]) & jnp.isfinite(z).all(axis=0)
                      & jnp.isfinite(h).all(axis=1) & jnp.isfinite(g).all(axis=1))
            masks = (better[:, None], better, better[None, :], better[:, None])
            return tuple(jnp.where(m, a, b) for m, a, b in zip(masks, (h, f, z, g), best, strict=True)), None

        best, _ = jax.lax.scan(take_best, state, jnp.asarray(STARTS, dtype=phi.dtype))
        return best

    return fit_restarts


def score(phi: np.ndarray, phit: np.ndarray, y: np.ndarray, yt: np.ndarray, chunk: int) -> np.ndarray:
    """Fit from prepared mean designs and training-scaled responses and return held-out z-scores."""
    d = design(phi, phit)
    _, _, z, _ = fit_chunks(make_fitter(d[0].shape[1]), d, y, yt, chunk)
    failed = np.flatnonzero(~np.isfinite(z).all(axis=0))
    if failed.size:
        raise ValueError(f"no finite ARD fit for response columns {failed.tolist()}")
    return z
