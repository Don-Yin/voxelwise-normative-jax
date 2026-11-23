"""Verify every saved flag, fraction, sample, count and slice against one named GPU receipt."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.artifacts import (
    PCNTOOLKIT_REFERENCE,
    VOXEL_MATRIX,
    resolve,
    sha256,
)
from shared.voxelwisenormative.batched_blr import N_ITER

from .model import (
    N_TRAIN,
    SURVIVAL_DENOMINATOR,
    SURVIVAL_NUMERATOR,
    TEST_ROW,
    Z_THRESHOLD,
    bootstrap_rows,
    survives,
)
from .slice_view import AXIAL_INDEX, N_VOXELS, raster, slice_columns

logger = logging.getLogger(__name__)


def verify(results: Path, run_name: str) -> dict:
    """Reject truncated, cross-run, inconsistent or incorrectly encoded evidence."""
    path = results / "json" / run_name / "result.json"
    report = json.loads(path.read_text())
    n = report["n_trials"]
    wanted = {"run_name": run_name, "n_voxels": N_VOXELS, "n_train": N_TRAIN, "n_iter": N_ITER,
              "test_row_zero_based": TEST_ROW, "z_threshold_abs_strict": Z_THRESHOLD,
              "survival_fraction_at_least": SURVIVAL_NUMERATOR / SURVIVAL_DENOMINATOR}
    for key, expected in wanted.items():
        if report[key] != expected:
            raise ValueError(f"{path}: {key}={report[key]}, expected {expected}")
    if len(report["trials"]) != n or report["slice"]["index_zero_based"] != AXIAL_INDEX:
        raise ValueError(f"{path}: inconsistent {n} trials or wrong slice {report['slice']}")
    payload = results / "arrays" / run_name / "bootstrap.npz"
    digest = sha256(payload)
    if (str(payload.relative_to(results)), digest) != (report["arrays"]["path"], report["arrays"]["sha256"]):
        raise ValueError(f"{payload}: digest {digest} or path differs from {path}")
    with np.load(resolve(PCNTOOLKIT_REFERENCE, "arrays.npz")["arrays.npz"]) as ref:
        train = np.flatnonzero(~ref["test"])
        held_out = int(np.flatnonzero(ref["test"])[TEST_ROW])
        if report["held_out_cohort_row_zero_based"] != held_out:
            raise ValueError(f"{path}: held-out cohort row differs from fixed row {held_out}")
    columns = slice_columns(resolve(VOXEL_MATRIX, "mask.nii.gz")["mask.nii.gz"])
    slice_ids = columns[columns >= 0]
    with np.load(payload, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["slice_columns"], columns)
        np.testing.assert_array_equal(data["reference_cohort_rows"], train)
        samples = train[bootstrap_rows(report["seed"], n, N_TRAIN)]
        np.testing.assert_array_equal(data["sampled_cohort_rows"], samples)
        z = data["baseline_z"]
        if z.shape != (N_VOXELS,) or not np.isfinite(z).all():
            raise ValueError(f"{payload}: invalid baseline shape {z.shape} or nonfinite scores")
        base = np.abs(z) > Z_THRESHOLD
        np.testing.assert_array_equal(data["reference_rgb"], raster(columns, z))
        frames = data["trial_slice_rgb"]
        np.testing.assert_array_equal(frames[0], raster(columns, z, base))
        np.testing.assert_equal((base.sum(), base[slice_ids].sum()),
                                (report["baseline"]["flagged_whole_mask"], report["baseline"]["flagged_slice"]))
        flags = np.unpackbits(data["packed_trial_flags"], axis=1, bitorder="little", count=N_VOXELS)
        fraction = data["running_survival_fraction"]
        np.testing.assert_equal(flags.shape, (n, N_VOXELS))
        np.testing.assert_equal(fraction.shape, flags.shape)
        np.testing.assert_equal(len(frames), n+1)
        counts = np.zeros(N_VOXELS, dtype=np.int32)
        for i, row in enumerate(report["trials"], 1):
            counts += flags[i-1]
            np.testing.assert_array_equal((counts / i).astype(np.float32), fraction[i-1])
            stable = survives(base, counts, i)
            np.testing.assert_equal((row["trial"], row["flagged_whole_mask"], row["surviving_whole_mask"],
                                     row["surviving_slice"], row["unique_reference_rows"]),
                                    (i, flags[i-1].sum(), stable.sum(), stable[slice_ids].sum(),
                                     np.unique(samples[i-1]).size))
            np.testing.assert_array_equal(frames[i], raster(columns, z, stable))
        flipped = data["opposite_sign_flag_count"]
        np.testing.assert_equal(flipped.shape, counts.shape)
        np.testing.assert_equal(np.any((flipped < 0) | (flipped > counts) | ((~base) & (flipped > 0))), False)
        np.testing.assert_equal(np.count_nonzero(flipped), report["baseline_flags_with_any_opposite_sign_trial"])
    logger.info("verified all %d trial masks, running fraction vectors, sampled cohorts and axial views: %s", n, path)
    return report
