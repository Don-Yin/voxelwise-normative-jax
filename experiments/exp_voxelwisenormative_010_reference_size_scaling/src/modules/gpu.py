"""Whole-mask timings of the early-stop batched fits; no automatic CPU fallback."""

from __future__ import annotations

import logging
import statistics
import time
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.batched_blr import N_ITER
from shared.voxelwisenormative.receipts import peak_rss_bytes, publish, runtime
from shared.voxelwisenormative.reference_cohort import load_cohort
from shared.voxelwisenormative.reference_fit import Z_TOLERANCE, chunk_width, memory, open_fitter, prepare, scores
from shared.voxelwisenormative.reference_grid import MASK_VOXELS, Cell

logger = logging.getLogger(__name__)
WARM_REPEATS = 3


def benchmark(cell: Cell, precision: str, out: Path, *, smoke: bool = False) -> Path:
    """Measure a warm-up and three full fits in a fresh process at one N and precision."""
    logger.info("start batched gpu %s precision=%s smoke=%s output=%s", cell.label, precision, smoke, out)
    start = time.perf_counter()
    kernel, device, jax_version = open_fitter(smoke=smoke)
    cohort = load_cohort(cell)
    input_seconds = time.perf_counter() - start
    voxels = np.arange(32 if smoke else MASK_VOXELS)
    width = min(chunk_width(cell.n), 16 if smoke else MASK_VOXELS)
    start = time.perf_counter()
    data = prepare(cohort, cohort.train, voxels, cohort.basis(), precision, width)
    preparation_seconds = time.perf_counter() - start
    logger.info("prepared %d train, %d test, %d voxels, chunk=%d in %.2fs", *data.y.shape[:1], len(cohort.test),
                len(voxels), width, preparation_seconds)
    times, errors = [], []
    repeats = 1 if smoke else WARM_REPEATS
    for i in range(repeats + 1):
        z, elapsed = scores(kernel, device, data, width)
        if cohort.reference is not None:
            error = cohort.reference.max_delta(voxels, z)
            if error > Z_TOLERANCE:
                raise ValueError(f"{cell.label} {precision} repeat {i} max |Δz| {error:g} exceeds {Z_TOLERANCE}")
            errors.append(error)
        times.append(elapsed)
        logger.info("[%d/%d] %s %s %s %.3fs", i + 1, repeats + 1, cell.label, precision,
                    "warmup" if i == 0 else "timed", elapsed)
        del z
    mem = memory(device)
    if not smoke and not mem["peak_bytes_in_use"]:
        raise RuntimeError(f"GPU allocator did not report peak live bytes on {device}: {mem}")
    return publish(out, {"schema_version": 1, "workload": "batched-early", "dataset": cell.dataset,
                         "n": cell.n, "n_test": len(cohort.test), "columns": len(voxels), "precision": precision,
                         "device": device.device_kind, "jax": jax_version, "chunk": width, "max_steps": N_ITER,
                         "input_seconds": input_seconds, "preparation_seconds": preparation_seconds,
                         "warmup_seconds": times[0], "seconds_repeats": times[1:], "seconds_median": statistics.median(times[1:]),
                         "host_peak_rss_bytes": peak_rss_bytes(), "pinned_z_max_abs_delta": max(errors) if errors else None,
                         **mem, **runtime()})
