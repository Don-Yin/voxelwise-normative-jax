"""Train-only, site-stratified refits; publish only aggregate flag recurrence."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .receipts import peak_rss_bytes, publish, runtime
from .reference_cohort import bootstrap_draw, load_cohort
from .reference_fit import Z_TOLERANCE, chunk_width, memory, open_fitter, prepare, scores
from .reference_grid import CHAINS, MASK_VOXELS, MIN_RECUR, TRIALS, Z_THRESHOLD, Cell

logger = logging.getLogger(__name__)
BOOTSTRAP_SEED = 20261124


def instability(baseline: np.ndarray, recurrence: np.ndarray) -> dict:
    """Count only original flags, with fewer than 52 recurrences marked unstable."""
    if baseline.shape != recurrence.shape or baseline.ndim != 2 or np.any(recurrence > TRIALS):
        raise ValueError(f"invalid baseline/recurrence arrays: {baseline.shape=}, {recurrence.shape=}")
    flags_by_person = baseline.sum(axis=1)
    unstable_by_person = (baseline & (recurrence < MIN_RECUR)).sum(axis=1)
    flags, unstable = int(flags_by_person.sum()), int(unstable_by_person.sum())
    valid = flags_by_person > 0
    if not flags:
        raise ValueError(f"no original |z|>{Z_THRESHOLD} flags; instability fraction is undefined")
    fractions = unstable_by_person[valid] / flags_by_person[valid]
    return {"baseline_flags": flags, "unstable_flags": unstable, "unstable_fraction": unstable / flags,
            "held_out_people_with_flags": int(valid.sum()),
            "per_person_fraction_p10_p50_p90": np.quantile(fractions, (0.1, 0.5, 0.9)).tolist()}


def worker(
    cell: Cell, chain: int, out: Path, *, smoke: bool = False,
    summarize_magnitudes: Callable[[np.ndarray, np.ndarray, np.ndarray], list[dict]] | None = None,
) -> Path:
    """Stream 64 refits; optionally summarise the original |z| values at flagged positions."""
    logger.info("start bootstrap %s chain=%d smoke=%s output=%s", cell.label, chain, smoke, out)
    kernel, device, version = open_fitter(smoke=smoke)
    cohort = load_cohort(cell, chain)
    voxels = np.arange(32 if smoke else MASK_VOXELS)
    width = min(16, chunk_width(cell.n)) if smoke else chunk_width(cell.n)
    trials = 2 if smoke else TRIALS
    basis = cohort.basis()
    rng = np.random.default_rng([BOOTSTRAP_SEED, CHAINS[chain], cell.n])
    counts = np.zeros((len(cohort.test), len(voxels)), dtype=np.uint8)
    baseline = original_abs_flags = None
    timings = []
    pinned_error = None
    for i in range(trials + 1):
        rows = cohort.train if i == 0 else bootstrap_draw(cohort, rng)
        start = time.perf_counter()
        data = prepare(cohort, rows, voxels, basis, "float64", width)
        preparation_seconds = time.perf_counter() - start
        z, seconds = scores(kernel, device, data, width)
        flags = np.abs(z) > Z_THRESHOLD
        if i == 0:
            baseline = flags
            if summarize_magnitudes is not None:
                original_abs_flags = np.abs(z[flags])
            if cohort.reference is not None:
                pinned_error = cohort.reference.max_delta(voxels, z)
                if pinned_error > Z_TOLERANCE:
                    raise ValueError(f"{cell.label} bootstrap baseline max |Δz| {pinned_error:g} exceeds {Z_TOLERANCE}")
        else:
            counts += flags
        timings.append({"refit": i, "fit_score_seconds": seconds, "preparation_seconds": preparation_seconds,
                        "unique_reference_rows": int(np.unique(rows).size)})
        logger.info("[%d/%d] n=%d chain=%d prepare=%.3fs, fit=%.3fs, flags=%d",
                    i + 1, trials + 1, cell.n, chain, preparation_seconds, seconds, flags.sum())
        del data, z
    if baseline is None:
        raise RuntimeError("bootstrap finished without an original fit")
    summary = ({"baseline_flags": int(baseline.sum()), "max_recurrence": int(counts.max())}
               if smoke else instability(baseline, counts))
    if summarize_magnitudes is not None:
        if original_abs_flags is None:
            raise RuntimeError("bootstrap did not retain the original flagged magnitudes")
        summary["baseline_abs_z_bins"] = summarize_magnitudes(baseline, original_abs_flags, counts)
    return publish(out, {"schema_version": 1, "workload": "bootstrap", "dataset": cell.dataset,
                         "n": cell.n, "n_test": len(cohort.test), "columns": len(voxels), "precision": "float64",
                         "device": device.device_kind, "jax": version, "chunk": width, "chain": chain,
                         "chain_seed": CHAINS[chain], "draw_seed": BOOTSTRAP_SEED, "trials": trials,
                         "z_abs_threshold": Z_THRESHOLD, "min_recurring_trials": MIN_RECUR if not smoke else None,
                         "pinned_z_max_abs_delta": pinned_error, "timings": timings,
                         "host_peak_rss_bytes": peak_rss_bytes(), **summary, **memory(device), **runtime()})
