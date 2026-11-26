"""batched warped BLR in JAX: PCNtoolkit 1.4.0's BLR for every voxel at once.

Model and objective are PCNtoolkit's:
  hyp = [log beta, epsilon, log b, log alpha]; warp z = sinh(b * (asinh(y) + epsilon))
  objective = negative log evidence - sum log warp'(y) + 0.1 * ||hyp||_2, starting at hyp = 1.
Reshaping: all voxels share the design Phi, so eig(Phi^T Phi) is computed once and each voxel's
evidence is one (D x N) @ (N x V) product plus elementwise work. The 4 hyperparameters are found
per voxel by damped Newton steps with exact JAX derivatives, vectorised over voxels.
The design enters the compiled fit as an argument, so resamples of equal size reuse one compile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PENALTY = 0.1
LOG2PI = 1.8378770664093453
N_ITER = 150  # reaches converged PCNtoolkit (L-BFGS-B eps 1e-7) on the 1,000-voxel reference


@dataclass(frozen=True)
class Design:
    """eigenbasis of the training design shared by every voxel."""

    lam: np.ndarray  # (D,) eigenvalues of Phi_train^T Phi_train
    G: np.ndarray  # (D, N_train) = (Phi_train Q)^T
    H: np.ndarray  # (N_test, D) = Phi_test Q


def design(Phi_train: np.ndarray, Phi_test: np.ndarray) -> Design:
    """rotate train and test designs into the eigenbasis of Phi_train^T Phi_train."""
    lam, Q = np.linalg.eigh(Phi_train.T @ Phi_train)
    return Design(np.clip(lam, 0.0, None), (Phi_train @ Q).T, Phi_test @ Q)


def make_fitter(n_iter: int = N_ITER):
    """jitted fit(lam, G, H, Y_train, Y_test) -> (hyp, objective, z_test) for one voxel chunk."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    def warp(y, eps, b):
        u = b * (jnp.arcsinh(y) + eps)
        return jnp.sinh(u), jnp.log(b * jnp.cosh(u)) - 0.5 * jnp.log1p(y**2)

    def posterior(hyp, z, lam, G):
        beta, alpha = jnp.exp(hyp[0]), jnp.exp(hyp[3])
        r = G @ z
        den = beta * lam + alpha
        return beta, alpha, r, den, beta * r / den  # posterior mean weights in the eigenbasis

    def objective(hyp, y, lam, G):
        N, D = y.shape[0], lam.shape[0]
        z, logjac = warp(y, hyp[1], jnp.exp(hyp[2]))
        beta, alpha, r, den, mt = posterior(hyp, z, lam, G)
        quad = z @ z - 2 * mt @ r + jnp.sum(lam * mt**2)
        nlz = 0.5 * (N * LOG2PI - N * hyp[0] - D * hyp[3] + beta * quad + alpha * mt @ mt + jnp.sum(jnp.log(den)))
        return nlz - jnp.sum(logjac) + PENALTY * jnp.sqrt(jnp.sum(hyp**2))

    def zscore(hyp, y, yt, lam, G, H):
        z, _ = warp(y, hyp[1], jnp.exp(hyp[2]))
        beta, _, _, den, mt = posterior(hyp, z, lam, G)
        zt, _ = warp(yt, hyp[1], jnp.exp(hyp[2]))
        s2 = 1.0 / beta + (H**2) @ (1.0 / den)
        return (zt - H @ mt) / jnp.sqrt(s2)

    axes = (0, 1, None, None)
    f = jax.vmap(objective, in_axes=axes)
    g = jax.vmap(jax.grad(objective), in_axes=axes)
    h = jax.vmap(jax.hessian(objective), in_axes=axes)

    @jax.jit
    def fit(lam, G, H, y, yt):
        def step(carry, _):
            hyp, mu, fv = carry
            g0, H0 = g(hyp, y, lam, G), h(hyp, y, lam, G)
            d = -jnp.linalg.solve(H0 + mu[:, None, None] * jnp.eye(4), g0[..., None])[..., 0]
            d = jnp.clip(d, -2.0, 2.0)
            f1 = f(hyp + d, y, lam, G)
            ok = jnp.isfinite(f1) & (f1 < fv)
            hyp = jnp.where(ok[:, None], hyp + d, hyp)
            mu = jnp.where(ok, jnp.maximum(mu / 3, 1e-9), mu * 4)
            return (hyp, mu, jnp.where(ok, f1, fv)), None

        hyp = jnp.ones((y.shape[1], 4))
        (hyp, _, fv), _ = jax.lax.scan(step, (hyp, jnp.full(y.shape[1], 1e-3), f(hyp, y, lam, G)), None, length=n_iter)
        z = jax.vmap(zscore, in_axes=(0, 1, 1, None, None, None), out_axes=1)(hyp, y, yt, lam, G, H)
        return hyp, fv, z

    return fit


def fit_chunks(fit, d: Design | tuple[np.ndarray, ...], Y: np.ndarray, Yt: np.ndarray, chunk: int = 32768,
               *, initial: np.ndarray | None = None):
    """Fit response columns in chunks; concatenate hyp, objective, z and any voxel-major diagnostics."""
    import jax

    if chunk < 1 or not Y.shape[1]:
        raise ValueError(f"need a positive chunk size and at least one voxel, got {chunk=} and {Y.shape=}")
    if initial is not None and (initial.ndim != 2 or initial.shape[0] != Y.shape[1]):
        raise ValueError(f"initial hyperparameters must be voxel-major for {Y.shape[1]} voxels, got {initial.shape}")
    arrays = (d.lam, d.G, d.H) if isinstance(d, Design) else d
    device_design = tuple(jax.device_put(a) for a in arrays)
    outs = []
    for s in range(0, Y.shape[1], chunk):
        sl = slice(s, s + chunk)
        responses = jax.device_put((Y[:, sl], Yt[:, sl]))
        starts = () if initial is None else (jax.device_put(initial[sl]),)
        outs.append(fit(*device_design, *responses, *starts))
    jax.block_until_ready(outs)
    return tuple(np.concatenate(columns, axis=1 if i == 2 else 0) for i, columns in enumerate(zip(*outs, strict=True)))
