"""Time the deployed ARD solver and use targeted restarts to explain voxelwise score differences."""

from __future__ import annotations

import json
import logging
import time
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np

from shared.voxelwisenormative import ard_blr
from shared.voxelwisenormative.batched_blr import PENALTY, fit_chunks
from shared.voxelwisenormative.pcn_cpu_agreement import (
    FIXED_HYP_TOL,
    OBJECTIVE_TOL,
    Z_TOL,
    agreement,
)
from shared.voxelwisenormative.pcn_cpu_reference import (
    EPSILON,
    IXIInputs,
    PCNReference,
    cpu_host,
)

logger = logging.getLogger(__name__)


def _z_tail(ref: PCNReference, objective: np.ndarray, z: np.ndarray) -> dict:
    """Count objective wins and losses specifically among voxels that disagree in held-out z."""
    tail = np.abs(z - ref.z).max(axis=0) > Z_TOL
    delta = objective - ref.objective
    return {
        "jax_lower_objective": (tail & (delta < -OBJECTIVE_TOL)).sum().item(),
        "pcn_lower_objective": (tail & (delta > OBJECTIVE_TOL)).sum().item(),
        "objective_within_tolerance": (tail & (np.abs(delta) <= OBJECTIVE_TOL)).sum().item(),
    }


def _audit_modes(data: IXIInputs, ref: PCNReference, d, evaluate, ones, pcn_seeded, deployed) -> dict:
    """Test curvature on both sides of the largest single-start score discrepancy."""
    col = np.abs(ones[2] - ref.z).max(axis=0).argmax().item()
    y, yt = data.y[:, col:col + 1], data.yt[:, col:col + 1]

    def objective(h):
        """Differentiate one scalar loss with respect to that voxel's parameters."""
        return evaluate(h[None, :], *d, y, yt)[0][0]

    hessian = jax.jit(jax.hessian(objective))
    modes = {}
    for name, (h, loss, z, g) in (("all_ones", ones), ("pcn_seeded", pcn_seeded), ("deployed", deployed)):
        curvature = np.linalg.eigvalsh(np.asarray(hessian(h[col])))
        modes[name] = {
            "objective": loss[col].item(), "gradient_inf": np.abs(g[col]).max().item(),
            "hessian_min_eigenvalue": curvature.min().item(),
            "z_max_absdiff_vs_pcn": np.abs(z[:, col] - ref.z[:, col]).max().item(),
            "warp_location": h[col, 1].item(), "log_warp_scale": h[col, 2].item(),
        }
    return {"mask_voxel": data.voxels[col].item(), "pcn_objective": ref.objective[col].item(), "fits": modes}


def benchmark(data: IXIInputs, ref: PCNReference, results: Path, repeats: int, *, profiled: bool) -> None:
    """Write a paired objective audit, separate from synchronized fit-and-score timings."""
    host = cpu_host()
    if host["os"] != "Darwin" or host["jax_backend"] != "cpu":
        raise ValueError(f"this benchmark requires Mac CPU, got {host['os']} and {host['jax_backend']}")
    n = data.voxels.size
    n_weights = data.phi.shape[1]
    started = time.perf_counter()
    d = ard_blr.design(data.phi, data.phit)
    deploy = ard_blr.make_fitter(n_weights)
    precompute = time.perf_counter() - started

    def timed_fit():
        """Time fitting all starts and collecting full predictions from the device."""
        start = time.perf_counter()
        result = fit_chunks(deploy, d, data.y, data.yt, chunk=n)
        return result, time.perf_counter() - start

    fitted, first = timed_fit()
    warm = []
    for i in range(repeats):
        fitted, seconds = timed_fit()
        warm.append(seconds)
        logger.info("[%d/%d] jax fit and score: %.3f s", i + 1, repeats, seconds)
    first_fit, evaluate = ard_blr.make_kernels(n_weights)
    fixed_f, fixed_z = (np.asarray(a) for a in evaluate(ref.hyp, *d, data.y, data.yt))
    logger.info("audit independent all-ones and pcntoolkit-seeded fits for %d voxels", n)
    ones = fit_chunks(first_fit, d, data.y, data.yt, chunk=n)
    pcn_seeded = fit_chunks(first_fit, d, data.y, data.yt, chunk=n, initial=ref.hyp)
    checks = {}
    for name, fit in (("all_ones", ones), ("pcn_seeded", pcn_seeded), ("deployed", fitted)):
        checks[name] = agreement(ref, fit, fixed_f, fixed_z, data.voxels)
        checks[name]["z_disagreement_by_objective"] = _z_tail(ref, fit[1], fit[2])
        checks[name]["objective_jax_minus_pcn_max"] = (fit[1] - ref.objective).max().item()
    restart_gain = ones[1] - fitted[1]
    median = np.median(warm).item()
    summary = {
        "host": host, "profiled": profiled,
        "dataset": {
            "archive": str(data.archive), "n_voxels": n, "n_train": len(data.y), "n_test": len(data.yt),
            "mean_design_columns": n_weights, "mean_design_rank": np.linalg.matrix_rank(data.phi).item(),
        },
        "pcntoolkit": {
            "version": version("pcntoolkit"), "total_seconds": ref.seconds,
            "fit_seconds_per_voxel_median": np.median(ref.fit_seconds).item(),
            "score_seconds_per_voxel_median": np.median(ref.score_seconds).item(),
            "design_max_absdiff": ref.design_error, "l_bfgs_b_epsilon": EPSILON, "penalty": PENALTY,
            "warnings": ref.warnings,
        },
        "jax_cpu": {
            "first_call_seconds": first, "precompute_seconds": precompute,
            "first_fit_with_precompute_seconds": first + precompute,
            "warm_seconds": warm, "warm_seconds_median": median, "warm_speedup_over_pcntoolkit": ref.seconds / median,
            "max_iters_per_start": ard_blr.MAX_ITERS, "starts": ard_blr.STARTS, "dtype": fitted[0].dtype.name,
        },
        "agreement": checks["deployed"],
        "initialization_audit": {
            "all_ones": checks["all_ones"], "pcn_seeded": checks["pcn_seeded"],
            "deployed_voxels_improving_all_ones": (restart_gain > OBJECTIVE_TOL).sum().item(),
            "deployed_max_improvement_over_all_ones": restart_gain.max().item(),
            "deployed_max_worsening_over_all_ones": np.maximum(0, -restart_gain.min()).item(),
            "worst_all_ones_voxel": _audit_modes(data, ref, d, evaluate, ones, pcn_seeded, fitted),
        },
    }
    table = results / "tables" / f"fit-arrays-{n}.npz"
    np.savez_compressed(
        table, allow_pickle=False, voxels=data.voxels,
        all_ones_hyp=ones[0], all_ones_objective=ones[1], all_ones_z=ones[2], all_ones_gradient=ones[3],
        pcn_seeded_hyp=pcn_seeded[0], pcn_seeded_objective=pcn_seeded[1], pcn_seeded_z=pcn_seeded[2],
        pcn_seeded_gradient=pcn_seeded[3], jax_hyp=fitted[0], jax_objective=fitted[1], jax_z=fitted[2],
        jax_gradient=fitted[3], jax_objective_at_pcn_params=fixed_f, jax_z_at_pcn_params=fixed_z,
    )
    path = results / "json" / f"benchmark-{n}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    logger.info("wrote %s and %s; %d/%d score-discrepant voxels at |Δz| > %s", table, path,
                checks["deployed"]["voxels_gt_z_tolerance"], n, Z_TOL)
    if not checks["deployed"]["same_hyp_passed"] or restart_gain.min() < -FIXED_HYP_TOL:
        raise ValueError(f"fixed-model parity or best-evidence selection failed; see {path}")
