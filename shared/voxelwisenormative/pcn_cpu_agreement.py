"""Compare paired JAX and PCNtoolkit warped fits and fixed-parameter predictions."""
from __future__ import annotations

import numpy as np

from .batched_solve import GRAD_TOL
from .pcn_cpu_reference import PCNReference

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
