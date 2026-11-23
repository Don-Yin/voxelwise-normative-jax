"""Equivalent warped BLR evidence and its explicit four-parameter derivatives."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from shared.voxelwisenormative.batched_blr import LOG2PI, PENALTY


def projector(tf32: bool):
    """select full mantissa or one TensorFloat-32 tensor-core dot product."""
    precision = jax.lax.DotAlgorithmPreset.TF32_TF32_F32 if tf32 else jax.lax.Precision.HIGHEST
    return partial(jnp.dot, precision=precision)


def stable_terms(hyp, x, lam):
    """return warp input and posterior factors in the rotated design basis."""
    p, eps, log_b, log_a = hyp
    b = jnp.exp(log_b)
    u = b * (x + eps)
    ratio = jnp.exp(log_a - p)
    inv = 1 / (lam + ratio)
    # log1p preserves the determinant's small terms when alpha dominates.
    logdet = jnp.sum(jnp.log1p(lam / ratio))
    return b, u, ratio, inv, logdet


def objective(hyp, x, lam, G, project):
    """return the evidence omitting only the terms constant in hyp."""
    _, u, _, inv, logdet = stable_terms(hyp, x, lam)
    z = jnp.sinh(u)
    r = project(G, z)
    quad = jnp.sum(z**2) - jnp.sum(inv * r**2)
    jac = x.size * hyp[2] + jnp.sum(jnp.log(jnp.cosh(u)))
    return 0.5 * (jnp.exp(hyp[0]) * quad - x.size * hyp[0] + logdet) - jac + PENALTY * jnp.linalg.norm(hyp)


def evidence_derivatives(hyp, x, lam, G, project):
    """explicit gradient and Hessian using three projections and one backprojection."""
    b, u, ratio, inv, _ = stable_terms(hyp, x, lam)
    z, c = jnp.sinh(u), jnp.cosh(u)
    z_e, z_b = b * c, u * c
    r, r_e, r_b = project(G, jnp.stack((z, z_e, z_b), axis=1)).T
    residual = z - project(G.T, inv * r)

    # q = z'z - r' inv r; v = log(alpha/beta).
    w = ratio * inv
    wr = w * inv * r
    q = jnp.sum(z * residual)
    q_v = jnp.sum(wr * r)
    q_vv = jnp.sum(wr * (1 - 2 * w) * r)
    q_e = 2 * jnp.sum(residual * z_e)
    q_b = 2 * jnp.sum(residual * z_b)
    q_ve = 2 * jnp.sum(wr * r_e)
    q_vb = 2 * jnp.sum(wr * r_b)
    q_ee = 2 * (jnp.sum(z_e**2) - jnp.sum(inv * r_e**2) + b**2 * q)
    q_eb = 2 * (jnp.sum(z_e * z_b + residual * (z_e + b * u * z)) - jnp.sum(inv * r_e * r_b))
    q_bb = 2 * (jnp.sum(z_b**2 + residual * (z_b + u**2 * z)) - jnp.sum(inv * r_b**2))

    logdet_v = -jnp.sum(lam * inv)
    logdet_vv = jnp.sum(w * lam * inv)
    tanh = z / c
    sech2 = 1 / c**2
    jac_e = b * jnp.sum(tanh)
    jac_b = x.size + jnp.sum(u * tanh)
    jac_ee = b**2 * jnp.sum(sech2)
    jac_eb = jac_e + b * jnp.sum(u * sech2)
    jac_bb = jac_b - x.size + jnp.sum(u**2 * sech2)

    half_beta = 0.5 * jnp.exp(hyp[0])
    gp = half_beta * (q - q_v) - 0.5 * (x.size + logdet_v)
    ga = half_beta * q_v + 0.5 * logdet_v
    grad = jnp.stack((gp, half_beta * q_e - jac_e, half_beta * q_b - jac_b, ga))
    pp = half_beta * (q - 2 * q_v + q_vv) + 0.5 * logdet_vv
    ee, eb, bb = half_beta * q_ee - jac_ee, half_beta * q_eb - jac_eb, half_beta * q_bb - jac_bb
    pa, aa = half_beta * (q_v - q_vv) - 0.5 * logdet_vv, half_beta * q_vv + 0.5 * logdet_vv
    pe, pb = half_beta * (q_e - q_ve), half_beta * (q_b - q_vb)
    ea, ba = half_beta * q_ve, half_beta * q_vb
    hess = jnp.stack((jnp.stack((pp, pe, pb, pa)), jnp.stack((pe, ee, eb, ea)),
                      jnp.stack((pb, eb, bb, ba)), jnp.stack((pa, ea, ba, aa))))
    norm = jnp.linalg.norm(hyp)
    return (grad + PENALTY * hyp / norm,
            hess + (PENALTY / norm) * (jnp.eye(4, dtype=hyp.dtype) - jnp.outer(hyp, hyp) / norm**2))


def predict(hyp, x, xt, lam, G, H, project):
    """predict warped test z-scores in the shared basis."""
    b, u, _, inv, _ = stable_terms(hyp, x, lam)
    mt = inv * project(G, jnp.sinh(u))
    zt = jnp.sinh(b * (xt + hyp[1]))
    s2 = jnp.exp(-hyp[0]) * (1 + project(H**2, inv))
    return (zt - project(H, mt)) / jnp.sqrt(s2)


def full_evidence(fv, y):
    """restore the objective constants for comparison with PCNtoolkit."""
    return fv + 0.5 * (y.shape[0] * LOG2PI + jnp.sum(jnp.log1p(y**2), axis=0))
