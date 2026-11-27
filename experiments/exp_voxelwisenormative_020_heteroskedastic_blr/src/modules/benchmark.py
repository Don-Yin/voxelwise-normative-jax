"""Compare heteroskedastic warped BLR fits on the frozen IXI reference, CPU only."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from shared.voxelwisenormative import heteroskedastic_blr
from shared.voxelwisenormative.pcn_cpu_agreement import agreement, report, time_fits
from shared.voxelwisenormative.pcn_cpu_reference import cpu_host, fit_pcn, load_ixi
from src.utils.paths import results_dir

logger = logging.getLogger(__name__)
MAX_ITERS = 300


def benchmark(n_voxels: int, repeats: int, *, root: Path | None, profiled: bool) -> None:
    """Measure one serial PCNtoolkit fit and repeated batched JAX fits on matching designs."""
    host = cpu_host()
    if host["jax_backend"] != "cpu":
        raise RuntimeError(f"the IXI timing requires the CPU, got {host['jax_backend']}")
    data = load_ixi(n_voxels, root=root)
    ref = fit_pcn(data, heteroskedastic=True)
    logger.info("pcntoolkit fit and score took %.3f s; start jax", ref.seconds)

    started = time.perf_counter()
    d = heteroskedastic_blr.design(data.phi, data.phit, ref.noise, ref.noiset)
    fit, evaluate = heteroskedastic_blr.make_kernels(ref.noise.shape[1], max_iters=MAX_ITERS)
    precompute = time.perf_counter() - started
    out, first, warm = time_fits(fit, d, data, repeats)
    fixed_objective, fixed_z = (np.asarray(a) for a in evaluate(ref.hyp, *d, data.y, data.yt))
    stats = agreement(ref, out, fixed_objective, fixed_z, data.voxels)
    summary = report(data, ref, host, profiled, out, first, precompute, warm,
                     dataset={"noise_design_columns": ref.noise.shape[1]}, jax_cpu={"max_iters": MAX_ITERS})
    summary["agreement"] = stats

    table_path = results_dir("tables") / f"fit-arrays-{n_voxels}.npz"
    np.savez_compressed(table_path, voxels=data.voxels, pcn_z=ref.z, pcn_hyp=ref.hyp,
                        pcn_objective=ref.objective, jax_hyp=out[0], jax_objective=out[1], jax_z=out[2],
                        jax_gradient=out[3], pcn_fit_seconds=ref.fit_seconds, pcn_score_seconds=ref.score_seconds)
    path = results_dir("json") / f"benchmark-{n_voxels}.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    logger.info("wrote %s and %s; max absolute z gap %.6g", path, table_path, stats["z_absdiff_max"])
    if not stats["same_hyp_passed"] or stats["voxels_gt_z_tolerance"]:
        raise RuntimeError(f"jax and pcntoolkit agreement failed: see {path}")
