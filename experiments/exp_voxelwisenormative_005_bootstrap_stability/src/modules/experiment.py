"""Materialize a verified whole-mask bootstrap, with the same real frames for live and replay."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.artifacts import (
    PCNTOOLKIT_REFERENCE,
    VOXEL_MATRIX,
    sha256,
)
from shared.voxelwisenormative.batched_blr import N_ITER

from .model import (
    SURVIVAL_DENOMINATOR,
    SURVIVAL_NUMERATOR,
    TEST_ROW,
    Z_THRESHOLD,
    bootstrap_rows,
    gpu_fitter,
    load_inputs,
    score,
    survives,
)
from .slice_view import AXIAL_INDEX, Z_LIMIT, raster

logger = logging.getLogger(__name__)


def save_results(results: Path, run_name: str, report: dict, **arrays) -> Path:
    """Publish arrays first and atomically publish a checksum-bound JSON completion marker."""
    payload = results / "arrays" / run_name / "bootstrap.npz"
    destination = results / "json" / run_name / "result.json"
    payload.parent.mkdir(parents=True, exist_ok=False)
    destination.parent.mkdir(parents=True, exist_ok=False)
    partial = payload.with_suffix(".tmp")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    partial.replace(payload)
    report["arrays"] = {"path": str(payload.relative_to(results)), "sha256": sha256(payload)}
    partial_json = destination.with_suffix(".tmp")
    partial_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    partial_json.replace(destination)
    logger.info("wrote %s", destination)
    return destination


def run(results: Path, run_name: str, trials: int, seed: int, chunk: int, publish) -> Path:
    """Fit the baseline and every trial; update the live display only for completed fits."""
    for kind in ("arrays", "json"):
        target = results / kind / run_name
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing run: {target}")
    total_start = time.perf_counter()
    fitter, jax_version, device = gpu_fitter()
    data = load_inputs()
    n_train, n_voxels = data.train.shape
    slice_voxels = data.columns[data.columns >= 0]
    samples = bootstrap_rows(seed, trials, n_train)
    logger.info("refitting %d voxels for %d trials on %s, held-out test row %d", n_voxels, trials, device, TEST_ROW)
    reference, baseline_fit = score(data, np.arange(n_train), fitter, chunk)
    baseline = np.abs(reference) > Z_THRESHOLD
    baseline_slice = int(baseline[slice_voxels].sum())
    reference_rgb = raster(data.columns, reference)
    slice_frames = [raster(data.columns, reference, baseline)]
    publish(0, baseline_slice, baseline_slice, reference_rgb, slice_frames[0])
    logger.info("baseline fit including compile: %.2f s, %d whole-mask flags, %d on the axial slice",
                baseline_fit, baseline.sum(), baseline_slice)

    flags = np.empty((trials, (n_voxels + 7) // 8), dtype=np.uint8)
    fraction = np.empty((trials, n_voxels), dtype=np.float32)
    count = np.zeros(n_voxels, dtype=np.int32)
    opposite = np.zeros(n_voxels, dtype=np.int32)
    rows = []
    for i, sample in enumerate(samples, 1):
        start = time.perf_counter()
        z, fit_seconds = score(data, sample, fitter, chunk)
        flagged = np.abs(z) > Z_THRESHOLD
        flags[i-1] = np.packbits(flagged, bitorder="little")
        count += flagged
        opposite += baseline & flagged & (np.signbit(z) != np.signbit(reference))
        fraction[i-1] = count / i
        stable = survives(baseline, count, i)
        stable_slice = int(stable[slice_voxels].sum())
        frame = raster(data.columns, reference, stable)
        slice_frames.append(frame)
        elapsed = time.perf_counter() - start
        rows.append({"trial": i, "unique_reference_rows": int(np.unique(sample).size),
                     "flagged_whole_mask": int(flagged.sum()), "surviving_whole_mask": int(stable.sum()),
                     "surviving_slice": stable_slice, "fit_seconds": fit_seconds, "seconds": elapsed})
        publish(i, stable_slice, baseline_slice, reference_rgb, frame)
        logger.info("[%d/%d] %.2f s (fit %.2f s), %d whole-mask survivors, %d/%d on slice",
                    i, trials, elapsed, fit_seconds, stable.sum(), stable_slice, baseline_slice)

    report = {
        "schema_version": 1, "run_name": run_name, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "device": device, "jax": jax_version, "seed": seed, "n_train": n_train, "n_voxels": n_voxels,
        "n_trials": trials, "n_iter": N_ITER, "chunk": chunk,
        "inputs": {"matrix": VOXEL_MATRIX, "split_and_design": PCNTOOLKIT_REFERENCE},
        "test_row_zero_based": TEST_ROW, "held_out_cohort_row_zero_based": data.held_out_cohort_row,
        "z_threshold_abs_strict": Z_THRESHOLD, "survival_fraction_at_least": SURVIVAL_NUMERATOR / SURVIVAL_DENOMINATOR,
        "bootstrap": "paired reference rows with replacement, fixed design basis, trial-local outcome scaling",
        "slice": {"orientation": "canonical RAS axial", "index_zero_based": AXIAL_INDEX,
                  "mask_voxels": int(slice_voxels.size), "z_colour_limits": [-Z_LIMIT, Z_LIMIT]},
        "baseline": {"flagged_whole_mask": int(baseline.sum()), "flagged_slice": baseline_slice,
                     "fit_seconds_including_compile": baseline_fit},
        "baseline_flags_with_any_opposite_sign_trial": int(np.count_nonzero(opposite)),
        "total_seconds_before_save": time.perf_counter() - total_start, "trials": rows,
        "trial_flags_encoding": {"key": "packed_trial_flags", "axis": 1, "bitorder": "little", "count": n_voxels},
    }
    return save_results(results, run_name, report, baseline_z=reference, packed_trial_flags=flags,
                        running_survival_fraction=fraction, opposite_sign_flag_count=opposite,
                        sampled_cohort_rows=data.train_cohort_rows[samples],
                        reference_cohort_rows=data.train_cohort_rows, slice_columns=data.columns,
                        reference_rgb=reference_rgb, trial_slice_rgb=np.stack(slice_frames))
