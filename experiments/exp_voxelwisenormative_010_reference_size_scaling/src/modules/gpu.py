"""Whole-mask timings of the early-stop batched fits; no automatic CPU fallback."""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.basis import Basis
from shared.voxelwisenormative.batched_blr import N_ITER, Design, design
from src.utils.measure import peak_rss_bytes, publish, runtime

from .cohorts import Cohort, load_cohort
from .grid import MASK_VOXELS, Cell

logger = logging.getLogger(__name__)
BLOCK_CAP = 32_768  # exp_004/006 at the original N=450
TRAIN_ELEMENTS_CAP = 450 * BLOCK_CAP
WARM_REPEATS = 3
Z_TOLERANCE = 0.01


@dataclass(frozen=True)
class Prepared:
    """Training-standardised data for all voxels, assembled before the fit timer."""

    rotated: Design
    y: np.ndarray
    yt: np.ndarray


def chunk_width(n: int) -> int:
    """Keep training elements per block bounded as reference size grows."""
    return min(BLOCK_CAP, max(1, TRAIN_ELEMENTS_CAP // n))


def prepare(cohort: Cohort, rows: np.ndarray, voxels: np.ndarray, basis: Basis, precision: str, chunk: int) -> Prepared:
    """Fit response scaling on this draw in float64, then store data at fit precision."""
    phi_train, phi_test = (basis.apply(cohort.x[sel], cohort.sites[sel]) for sel in (rows, cohort.test))
    rotated = design(phi_train, phi_test)
    # Row-major like the device arrays: a Fortran-order block costs ~0.4 s of transposing copies per whole-mask fit.
    y = np.empty((len(rows), len(voxels)), dtype=precision)
    yt = np.empty((len(cohort.test), len(voxels)), dtype=precision)
    for lo in range(0, len(voxels), chunk):
        cols = slice(lo, lo + chunk)
        train, test = cohort.responses(rows, voxels[cols])
        train = train.astype(np.float64)
        mean, sd = train.mean(axis=0), train.std(axis=0)
        bad = ~np.isfinite(mean) | ~np.isfinite(sd) | (sd <= 0)
        if bad.any():
            raise ValueError(f"{len(rows)} training rows give invalid scale at {bad.sum()} voxels in block; "
                             f"first mask column {voxels[lo + np.flatnonzero(bad)[0]]}")
        train -= mean
        train /= sd
        y[:, cols] = train
        yt[:, cols] = (test - mean) / sd
    return Prepared(rotated, y, yt)


def open_fitter(*, smoke: bool):
    """Use exactly the exp_004 analytic early-stop kernel in either precision."""
    import jax

    from shared.voxelwisenormative.early_newton import make_variant

    jax.config.update("jax_enable_x64", True)
    device = jax.devices("cpu" if smoke else "gpu")[0]
    if not smoke and (device.platform != "gpu" or "A100" not in device.device_kind):
        raise RuntimeError(f"benchmark requires an NVIDIA A100, got {device.device_kind} on {device.platform}")
    return make_variant(analytic=True, early=True, tf32=False), device, jax.__version__


def scores(kernel, device, data: Prepared, chunk: int) -> tuple[np.ndarray, float]:
    """Time a full synchronised fit, including transfers and gathering; then check output."""
    import jax

    start = time.perf_counter()
    dtype = data.y.dtype
    d = data.rotated
    prepared = tuple(jax.device_put(np.asarray(a, dtype=dtype), device) for a in (d.lam, d.G, d.H))
    n, v = data.y.shape
    window = 6 if n <= 450 else 2  # full queue at N=450; bound simultaneous buffers at high N
    z = np.empty_like(data.yt)
    evidence, parameters = [], []
    batch = []
    for lo in range(0, v, chunk):
        cols = slice(lo, min(lo + chunk, v))
        block = [jax.device_put(np.ascontiguousarray(array[:, cols]), device) for array in (data.y, data.yt)]
        batch.append((cols, kernel(*prepared, *block)))
        if len(batch) == window or lo + chunk >= v:
            jax.block_until_ready([result for _, result in batch])
            for cols, (hyp, objective, block_z, _) in batch:
                z[:, cols] = np.asarray(block_z)
                evidence.append(np.asarray(objective))
                parameters.append(np.asarray(hyp))
            batch.clear()
    seconds = time.perf_counter() - start
    for label, arrays in (("z", (z,)), ("evidence", evidence), ("hyperparameters", parameters)):
        if any(not np.isfinite(part).all() for part in arrays):
            raise ValueError(f"non-finite {label} for {n} training rows, {v} voxels, precision={dtype}")
    return z, seconds


def memory(device) -> dict[str, int | None]:
    """Allocator live bytes and reserved pool are different quantities."""
    stats = device.memory_stats() or {}
    keys = ("peak_bytes_in_use", "peak_pool_bytes", "bytes_limit")
    return {k: int(stats[k]) if k in stats else None for k in keys}


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
