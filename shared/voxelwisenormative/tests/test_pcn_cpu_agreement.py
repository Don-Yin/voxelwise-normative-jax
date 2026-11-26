"""Benchmark counts serialize without a NumPy-specific JSON encoder."""
import json

import numpy as np

from shared.voxelwisenormative.pcn_cpu_agreement import agreement
from shared.voxelwisenormative.pcn_cpu_reference import PCNReference


def test_paired_counts_are_json_serializable():
    hyp = np.ones((2, 5))
    objective = np.ones(2)
    z = np.zeros((3, 2))
    ref = PCNReference(hyp=hyp, objective=objective, z=z, noise=z, noiset=z,
                       fit_seconds=objective, score_seconds=objective, seconds=1.0,
                       design_error=0.0, warnings={})
    fit = hyp, objective + [-0.1, 0.1], z + [0.0, 0.1], np.zeros_like(hyp)
    report = json.loads(json.dumps(agreement(ref, fit, objective, z, np.array([40, 81])), allow_nan=False))
    assert report["n_test_scores"] == z.size
    assert report["voxels_gt_z_tolerance"] == 1 and report["worst_z_mask_voxel"] == 81
    assert report["voxels_jax_objective_better"] == report["voxels_jax_objective_worse"] == 1
    assert report["same_hyp_passed"] and report["jax_voxels_at_grad_tol"] == 2
