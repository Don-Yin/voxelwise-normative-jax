"""Time and device peak memory of one batched fit at N strided mask voxels; one process per case.

Variants are the two fits behind the deck's A100 numbers: `early-f32` is exp_voxelwisenormative_004's
fastest fit that passed its 0.01 z gate (promoted to shared/voxelwisenormative/early_newton.py), and
`baseline` is the shared 64-bit solver of exp_voxelwisenormative_003. Each timed fit includes host to
device transfers and gathering to host, as in those experiments. A fresh process per case makes the
allocator's `peak_bytes_in_use` the peak of that case alone.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import resource
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.batched_blr import Design

logger = logging.getLogger(__name__)

VARIANTS = {"early-f32": "float32", "baseline": "float64"}
TOL = 0.01  # exp_voxelwisenormative_004's reference z gate
MEMORY_KEYS = ("bytes_in_use", "peak_bytes_in_use", "pool_bytes", "peak_pool_bytes", "bytes_limit",
               "largest_alloc_size", "num_allocs")


@dataclass(frozen=True)
class Inputs:
    """scaled train/test columns, the shared design and the reference voxels inside this subset."""

    y: np.ndarray
    yt: np.ndarray
    design: Design
    columns: np.ndarray
    ref_positions: np.ndarray  # positions in `columns` of reference voxels
    ref_z: np.ndarray  # (n_test, len(ref_positions)) PCNtoolkit test z of those voxels


def strided_columns(n_voxels: int, n_mask: int) -> np.ndarray:
    """n evenly spaced mask columns in mask order; every column when n equals the mask size."""
    if not 1 <= n_voxels <= n_mask:
        raise ValueError(f"n_voxels must be within 1..{n_mask}: {n_voxels}")
    columns = np.round(np.linspace(0, n_mask - 1, n_voxels)).astype(np.int64)
    if np.unique(columns).size != n_voxels:
        raise RuntimeError(f"strided columns are not unique for n_voxels={n_voxels}")
    return columns


def load_ixi(n_voxels: int) -> Inputs:
    """verified IXI grey matter at strided columns, scaled as PCNtoolkit, with the frozen split and design."""
    from shared.voxelwisenormative.artifacts import PCNTOOLKIT_REFERENCE, resolve
    from shared.voxelwisenormative.batched_blr import design
    from shared.voxelwisenormative.cohort import load_grey_matter, standardise

    with np.load(resolve(PCNTOOLKIT_REFERENCE, "arrays.npz")["arrays.npz"]) as ref:
        test, ref_voxels = ref["test"], ref["voxels"]
        d = design(ref["Phi_train"], ref["Phi_test"])
        ref_train, ref_z = ref["y_train"], ref["z_test"]
    gm = load_grey_matter(mmap=True)
    columns = strided_columns(n_voxels, gm.shape[1])
    y, yt = standardise(np.asarray(gm[:, columns]), test)
    inside = np.isin(ref_voxels, columns)
    positions = np.searchsorted(columns, ref_voxels[inside])
    dy = float(np.abs(y[:, positions] - ref_train[:, inside]).max()) if positions.size else 0.0
    if dy > 1e-9:
        raise RuntimeError(f"standardisation differs from PCNtoolkit by {dy}")
    return Inputs(y, yt, d, columns, positions, ref_z[:, inside])


def make_kernel(variant: str):
    """the compiled fit for one variant; both reuse their programs when a chunk width repeats."""
    if variant == "baseline":
        from shared.voxelwisenormative.batched_blr import make_fitter

        return make_fitter()
    if variant == "early-f32":
        from shared.voxelwisenormative.early_newton import make_variant

        return make_variant(analytic=True, early=True, tf32=False)
    raise ValueError(f"unknown variant {variant}; options: {sorted(VARIANTS)}")


def fit_once(kernel, variant: str, data: Inputs, chunk: int, device):
    """one whole fit with transfers and host gathering; returns objective, test z and iteration counts."""
    import jax

    if variant == "baseline":
        from shared.voxelwisenormative.batched_blr import fit_chunks

        with jax.default_device(device):
            _, fv, z = fit_chunks(kernel, data.design, data.y, data.yt, chunk)
        return fv, z, None
    # exp_voxelwisenormative_004 dispatch.fit for one device: float32 design and blocks, mask order.
    host_design = tuple(np.asarray(a, dtype=np.float32) for a in (data.design.lam, data.design.G, data.design.H))
    prepared = tuple(jax.device_put(a, device) for a in host_design)
    outputs = []
    for lo in range(0, data.y.shape[1], chunk):
        cols = slice(lo, min(lo + chunk, data.y.shape[1]))
        y, yt = (jax.device_put(np.ascontiguousarray(a[:, cols], dtype=np.float32), device) for a in (data.y, data.yt))
        outputs.append(kernel(*prepared, y, yt))
    jax.block_until_ready(outputs)
    fv = np.concatenate([np.asarray(out[1]) for out in outputs])
    z = np.concatenate([np.asarray(out[2]) for out in outputs], axis=1)
    counts = np.concatenate([np.asarray(out[3]) for out in outputs])
    return fv, z, counts


def memory(device) -> dict:
    """the allocator counters this device reports; empty on CPU."""
    stats = device.memory_stats() or {}
    return {key: int(stats[key]) for key in MEMORY_KEYS if key in stats}


def host_peak_rss_bytes() -> int:
    """process lifetime peak resident memory (Linux reports KiB, macOS bytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def measure(variant: str, data: Inputs, chunk: int, repeats: int, device) -> dict:
    """one warm-up fit over every block width, then timed repeats and the allocator peak."""
    if repeats < 1 or chunk < 1:
        raise ValueError(f"repeats and chunk must be positive: {repeats=}, {chunk=}")
    kernel = make_kernel(variant)
    before = memory(device)
    start = time.perf_counter()
    fit_once(kernel, variant, data, chunk, device)
    warmup = time.perf_counter() - start
    after_warmup = memory(device)
    seconds = []
    fv = z = counts = None
    for i in range(repeats):
        start = time.perf_counter()
        fv, z, counts = fit_once(kernel, variant, data, chunk, device)
        seconds.append(time.perf_counter() - start)
        logger.info("[%d/%d] %s n=%d %.3f s", i + 1, repeats, variant, data.y.shape[1], seconds[-1])
    assert fv is not None and z is not None
    end = memory(device)
    dz = np.abs(z[:, data.ref_positions] - data.ref_z)
    nonfinite = int(z.size - np.isfinite(z).sum())
    max_dz = float(dz.max()) if dz.size else None
    widths = [min(chunk, data.y.shape[1] - lo) for lo in range(0, data.y.shape[1], chunk)]
    result = {
        "status": "measured",
        "warmup_seconds": warmup,
        "seconds": seconds,
        "seconds_median": statistics.median(seconds),
        "seconds_best": min(seconds),
        "block_widths": widths,
        "memory_before": before,
        "memory_after_warmup": after_warmup,
        "memory_end": end,
        "peak_bytes_in_use": end.get("peak_bytes_in_use"),
        "host_peak_rss_bytes": host_peak_rss_bytes(),
        "reference_voxels_in_subset": int(data.ref_positions.size),
        "z_absdiff_max": max_dz,
        "nonfinite_z": nonfinite,
        "gate_passed": nonfinite == 0 and (max_dz is None or max_dz <= TOL),
    }
    if counts is not None:
        result.update(iterations_p50=float(np.median(counts)), iterations_max=int(counts.max()))
    return result


def run(variant: str, n_voxels: int, chunk: int, repeats: int, require_gpu: bool, results: Path) -> dict:
    """measure one variant at one N on IXI data and write its JSON."""
    import jax

    jax.config.update("jax_enable_x64", True)  # as in exp_003 and exp_004; float32 inputs stay float32
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}; options: {sorted(VARIANTS)}")
    devices = jax.devices("gpu") if require_gpu else jax.devices()
    device = devices[0]
    data = load_ixi(n_voxels)
    job = os.environ.get("SLURM_JOB_ID")
    run_id = f"job-{job}" if job else f"local-{time.time_ns()}"
    path = results / "json" / "gpu" / run_id / f"{variant}-n{n_voxels:06d}.json"
    logger.info("%s n=%d on %s -> %s", variant, n_voxels, device.device_kind, path)
    result = {
        "mode": "gpu", "variant": variant, "dtype": VARIANTS[variant], "n_voxels": n_voxels,
        "n_mask": int(data.columns.max()) + 1 if n_voxels > 1 else None, "columns": "round(linspace(0, n_mask - 1, n))",
        "n_train": int(data.y.shape[0]), "n_test": int(data.yt.shape[0]), "chunk": chunk, "repeats": repeats,
        "device": device.device_kind, "platform": device.platform, "jax": jax.__version__,
        "python": platform.python_version(), "slurm_job_id": job, "result_path": str(path),
    }
    result.update(measure(variant, data, chunk, repeats, device))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    logger.info("%s", json.dumps({k: result[k] for k in ("variant", "n_voxels", "seconds_median", "peak_bytes_in_use",
                                                         "z_absdiff_max", "gate_passed")}))
    if not result["gate_passed"]:
        raise RuntimeError(f"{variant} n={n_voxels} failed the z gate or produced non-finite z: {path}")
    return result
