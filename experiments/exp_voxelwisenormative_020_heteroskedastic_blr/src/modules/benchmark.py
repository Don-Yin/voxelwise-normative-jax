"""Compare heteroskedastic warped BLR fits on the frozen IXI reference, CPU only."""
from __future__ import annotations

import json
import logging
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from shared.voxelwisenormative import heteroskedastic_blr
from shared.voxelwisenormative.batched_blr import PENALTY, fit_chunks
from shared.voxelwisenormative.pcn_cpu_agreement import agreement
from shared.voxelwisenormative.pcn_cpu_reference import (
    EPSILON,
    cpu_host,
    fit_pcn,
    load_ixi,
)
from src.utils.paths import results_dir

logger = logging.getLogger(__name__)
MAX_ITERS = 300


def benchmark(n_voxels: int, repeats: int, *, root: Path | None, profiled: bool) -> None:
    """Measure one serial PCNtoolkit fit and repeated batched JAX fits on matching designs."""
    host = cpu_host()
    if host["jax_backend"] != "cpu":
        raise RuntimeError(f"the IXI timing requires the CPU, got {host['jax_backend']}")
    if repeats < 1:
        raise ValueError(f"repeats must be positive, got {repeats}")
    data = load_ixi(n_voxels, root=root)
    ref = fit_pcn(data, heteroskedastic=True)
    logger.info("pcntoolkit fit and score took %.3f s; start jax", ref.seconds)

    started = time.perf_counter()
    d = heteroskedastic_blr.design(data.phi, data.phit, ref.noise, ref.noiset)
    fit, evaluate = heteroskedastic_blr.make_kernels(ref.noise.shape[1], max_iters=MAX_ITERS)
    design_seconds = time.perf_counter() - started

    def timed_fit():
        """Time one synchronized fit, including result collection."""
        start = time.perf_counter()
        result = fit_chunks(fit, d, data.y, data.yt, chunk=n_voxels)
        return result, time.perf_counter() - start

    out, first = timed_fit()
    logger.info("jax first fit and score (with compilation) took %.3f s", first)
    seconds = []
    for i in range(repeats):
        out, elapsed = timed_fit()
        seconds.append(elapsed)
        logger.info("[%d/%d] jax fit and score took %.3f s", i + 1, repeats, elapsed)
    fixed_objective, fixed_z = (np.asarray(a) for a in evaluate(ref.hyp, *d, data.y, data.yt))
    stats = agreement(ref, out, fixed_objective, fixed_z, data.voxels)

    median = float(np.median(seconds))
    report = {
        "host": host,
        "profiled": profiled,
        "dataset": {
            "archive": str(data.archive),
            "n_voxels": n_voxels,
            "n_train": len(data.y),
            "n_test": len(data.yt),
            "mean_design_columns": data.phi.shape[1],
            "mean_design_rank": int(np.linalg.matrix_rank(data.phi)),
            "noise_design_columns": ref.noise.shape[1],
        },
        "pcntoolkit": {
            "version": version("pcntoolkit"),
            "total_seconds": ref.seconds,
            "fit_seconds_per_voxel_median": float(np.median(ref.fit_seconds)),
            "score_seconds_per_voxel_median": float(np.median(ref.score_seconds)),
            "design_max_absdiff": ref.design_error,
            "l_bfgs_b_epsilon": EPSILON,
            "penalty": PENALTY,
            "warnings": ref.warnings,
        },
        "jax_cpu": {
            "first_call_seconds": first,
            "precompute_seconds": design_seconds,
            "first_fit_with_precompute_seconds": first + design_seconds,
            "warm_seconds": seconds,
            "warm_seconds_median": median,
            "warm_speedup_over_pcntoolkit": ref.seconds / median,
            "max_iters": MAX_ITERS,
            "dtype": out[0].dtype.name,
        },
        "agreement": stats,
    }
    table_path = results_dir("tables") / f"fit-arrays-{n_voxels}.npz"
    np.savez_compressed(table_path, voxels=data.voxels, pcn_z=ref.z, pcn_hyp=ref.hyp,
                        pcn_objective=ref.objective, jax_hyp=out[0], jax_objective=out[1], jax_z=out[2],
                        jax_gradient=out[3], pcn_fit_seconds=ref.fit_seconds, pcn_score_seconds=ref.score_seconds)
    path = results_dir("json") / f"benchmark-{n_voxels}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    logger.info("wrote %s and %s; max absolute z gap %.6g", path, table_path, stats["z_absdiff_max"])
    if not stats["same_hyp_passed"] or stats["voxels_gt_z_tolerance"]:
        raise RuntimeError(f"jax and pcntoolkit agreement failed: see {path}")
