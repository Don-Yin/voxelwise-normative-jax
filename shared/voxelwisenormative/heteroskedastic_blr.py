"""Warped BLR with a fitted noise-precision curve for each response column."""
from __future__ import annotations

import numpy as np
from jax import numpy as jnp

from . import batched_solve


def variance_basis(x, mean, sd, variance_design=None) -> np.ndarray:
    """Use training-scaled linear inputs and an intercept, or validate a supplied design."""
    if variance_design is None:
        return np.column_stack(((np.asarray(x, dtype=np.float64) - mean) / sd, np.ones(len(x))))
    noise = np.asarray(variance_design, dtype=np.float64)
    if (noise.ndim != 2 or len(noise) != len(x) or not np.isfinite(noise).all()
            or not np.any(np.all(noise == 1, axis=0))):
        raise ValueError(f"variance_design must be finite, have {len(x)} rows and include a column of ones; "
                         f"got {noise.shape}")
    return noise


def design(phi, phit, noise, noiset) -> tuple[np.ndarray, ...]:
    """Precompute flattened training subject outer products, shared across all voxels."""
    phi, phit, noise, noiset = (np.asarray(a, dtype=np.float64) for a in (phi, phit, noise, noiset))
    if any(a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all() for a in (phi, phit, noise, noiset)):
        raise ValueError(f"mean and noise designs must be nonempty, finite matrices: "
                         f"{phi.shape=}, {phit.shape=}, {noise.shape=}, {noiset.shape=}")
    if (len(phi) != len(noise) or len(phit) != len(noiset) or
            phi.shape[1] != phit.shape[1] or noise.shape[1] != noiset.shape[1]):
        raise ValueError(f"mean and noise designs disagree between train and test: "
                         f"{phi.shape=}, {phit.shape=}, {noise.shape=}, {noiset.shape=}")
    products = (phi[:, :, None] * phi[:, None, :]).reshape(len(phi), -1)
    return phi, phit, noise, noiset, products


def _noise_weights(w, phi, noise, products):
    """Compute all weighted Gram matrices with one matrix multiplication."""
    log_beta = w @ noise.T
    d = phi.shape[1]
    return log_beta, (jnp.exp(log_beta) @ products).reshape(-1, d, d)


def make_kernels(n_beta: int, *, max_iters: int = 300):
    """Build Cholesky kernels with one prior precision and n_beta noise coefficients."""
    return batched_solve.make_kernels(n_beta, 1, _noise_weights, max_iters=max_iters)
