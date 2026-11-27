"""Compare paired JAX and PCNtoolkit warped fits and fixed-parameter predictions."""
from __future__ import annotations

import logging
import time
from importlib.metadata import version

import numpy as np

from .batched_blr import PENALTY, fit_chunks
from .batched_solve import GRAD_TOL
from .pcn_cpu_reference import EPSILON, IXIInputs, PCNReference

logger = logging.getLogger(__name__)

Z_TOL = 0.01
FIXED_HYP_TOL = 1e-7
OBJECTIVE_TOL = 1e-3


def agreement(reference: PCNReference, fit_tuple, fixed_objective, fixed_z, voxels: np.ndarray) -> dict:
    """Separate numerical agreement of the model from agreement of the two optimizers."""
    hyp, objective, z, grad = map(np.asarray, fit_tuple)
    fixed_objective, fixed_z = map(np.asarray, (fixed_objective, fixed_z))
    for name, array, shape in (
        ("hyperparameters", hyp, reference.hyp.shape), ("gradient", grad, reference.hyp.shape),
        ("objective", objective, reference.objective.shape), ("z", z, reference.z.shape),
        ("fixed-parameter objective", fixed_objective, reference.objective.shape),
        ("fixed-parameter z", fixed_z, reference.z.shape),
    ):
        if array.shape != shape:
            raise ValueError(f"{name}: observed shape {array.shape}, expected {shape}")
        bad = np.count_nonzero(~np.isfinite(array))
        if bad:
            raise ValueError(f"{name}: {bad} nonfinite values in shape {array.shape}")

    dz = np.abs(z - reference.z)
    df = objective - reference.objective
    max_by_voxel = dz.max(axis=0)
    fixed_z_error = np.abs(fixed_z - reference.z).max().item()
    fixed_f_error = np.abs(fixed_objective - reference.objective).max().item()
    worsen = df > OBJECTIVE_TOL
    improve = df < -OBJECTIVE_TOL
    grad_max = np.abs(grad).max(axis=1)
    return {
        "n_test_scores": dz.size,
        "same_hyp_z_max_absdiff": fixed_z_error,
        "same_hyp_objective_max_absdiff": fixed_f_error,
        "same_hyp_tolerance": FIXED_HYP_TOL,
        "same_hyp_passed": max(fixed_z_error, fixed_f_error) <= FIXED_HYP_TOL,
        "z_absdiff_median": np.median(dz).item(),
        "z_absdiff_p99": np.quantile(dz, 0.99).item(),
        "z_absdiff_max": dz.max().item(),
        "z_tolerance": Z_TOL,
        "z_fraction_within_tolerance": np.mean(dz <= Z_TOL).item(),
        "voxels_gt_z_tolerance": int(np.count_nonzero(max_by_voxel > Z_TOL)),
        "worst_z_mask_voxel": voxels[max_by_voxel.argmax()].item(),
        "objective_tolerance": OBJECTIVE_TOL,
        "objective_jax_minus_pcn_median": np.median(df).item(),
        "voxels_jax_objective_better": int(np.count_nonzero(improve)),
        "voxels_jax_objective_worse": int(np.count_nonzero(worsen)),
        "jax_grad_tol": GRAD_TOL,
        "jax_voxels_at_grad_tol": int(np.count_nonzero(grad_max <= GRAD_TOL)),
        "jax_final_gradient_inf_max": grad_max.max().item(),
    }


def time_fits(fit, d, data: IXIInputs, repeats: int):
    """Time one compiling call and `repeats` warm calls, each including result collection."""
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    seconds = []
    for i in range(repeats + 1):
        start = time.perf_counter()
        out = fit_chunks(fit, d, data.y, data.yt, chunk=data.voxels.size)
        seconds.append(time.perf_counter() - start)
        logger.info("[%d/%d] jax fit and score: %.3f s%s", i, repeats, seconds[-1], " (compiling)" * (i == 0))
    return out, seconds[0], seconds[1:]


def report(data: IXIInputs, ref: PCNReference, host: dict, profiled: bool, out, first: float,
           precompute: float, warm: list[float], *, dataset: dict, jax_cpu: dict) -> dict:
    """Shared host, input, PCNtoolkit and JAX timing sections of a CPU benchmark report."""
    median = np.median(warm).item()
    return {
        "host": host, "profiled": profiled,
        "dataset": {
            "archive": str(data.archive), "n_voxels": data.voxels.size, "n_train": len(data.y),
            "n_test": len(data.yt), "mean_design_columns": data.phi.shape[1],
            "mean_design_rank": np.linalg.matrix_rank(data.phi).item(), **dataset,
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
            "first_fit_with_precompute_seconds": first + precompute, "warm_seconds": warm,
            "warm_seconds_median": median, "warm_speedup_over_pcntoolkit": ref.seconds / median,
            "dtype": np.asarray(out[0]).dtype.name, **jax_cpu,
        },
    }
