"""Warm, time and gate every variant before selecting a chunk size and GPU count."""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from pathlib import Path

import jax
import numpy as np

from shared.voxelwisenormative.artifacts import PCNTOOLKIT_REFERENCE

from .dataset import TOL, agreement, load
from .dispatch import VARIANTS, fit, shards

logger = logging.getLogger(__name__)


def save_json(path: Path, payload: dict) -> None:
    """publish complete JSON only, including for a measured failed variant."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def measure(data, variant: str, chunk: int, devices, repeats: int) -> dict:
    """warm all shapes, then measure fit plus transfers, excluding validation and I/O."""
    logger.info("start variant=%s chunk=%d devices=%d", variant, chunk, len(devices))
    start = time.perf_counter()
    _, _, counts = fit(data, variant, chunk, devices)
    warmup = time.perf_counter() - start
    seconds = []
    accuracy = []
    for i in range(repeats):
        start = time.perf_counter()
        fv, z, counts = fit(data, variant, chunk, devices)
        seconds.append(time.perf_counter() - start)
        accuracy.append(agreement(data, fv, z))
        del fv, z
        logger.info("[%d/%d] %s chunk=%d %.3f s max dz=%s nonfinite=%d", i + 1, repeats, variant, chunk,
                    seconds[-1], accuracy[-1]["z_absdiff_max"], accuracy[-1]["nonfinite_z"])
    maxima = [entry["z_absdiff_max"] for entry in accuracy]
    sizes = [part.stop - part.start for _, part in shards(data.y.shape[1], chunk, devices)]
    return {
        "status": "measured",
        "warmup_seconds": warmup,
        "seconds": seconds,
        "seconds_best": min(seconds),
        "seconds_median": statistics.median(seconds),
        "z_absdiff_max": None if None in maxima else max(maxima),
        "nonfinite_z": max(entry["nonfinite_z"] for entry in accuracy),
        "gate_passed": all(entry["passed"] for entry in accuracy),
        "accuracy_by_repeat": accuracy,
        "block_widths": sizes,
        "iterations_p50": float(np.median(counts)),
        "iterations_p99": float(np.quantile(counts, 0.99)),
        "iterations_max": int(counts.max()),
        "voxel_steps": int(counts.sum()),
        "block_steps": int(np.maximum.reduceat(counts, np.r_[0, np.cumsum(sizes[:-1], dtype=int)]).sum()),
    }


def run_case(data, mode, variant, chunk, devices, repeats, out, job_id) -> dict:
    """persist a case result even when compilation or execution raises an error."""
    dtype, analytic, early, tf32 = VARIANTS[variant]
    path = out / f"{variant}-{chunk}" / "result.json"
    result = {
        "mode": mode, "variant": variant, "chunk": chunk, "gpu_count": sum(d.platform == "gpu" for d in devices),
        "devices": [d.device_kind for d in devices], "jax": jax.__version__, "slurm_job_id": job_id,
        "dtype": dtype.__name__, "analytic": analytic, "early": early, "tf32": tf32,
        "n_train": data.y.shape[0], "n_test": data.yt.shape[0], "n_voxels": data.y.shape[1],
        "check_voxels": data.idx.size, "reference": PCNTOOLKIT_REFERENCE, "z_tolerance": TOL,
        "result_path": str(path),
    }
    try:
        result.update(measure(data, variant, chunk, devices, repeats))
    except (RuntimeError, NotImplementedError, ValueError) as exc:
        logger.exception("variant=%s chunk=%d failed", variant, chunk)
        result.update(status="error", error=str(exc), gate_passed=False,
                      seconds_best=None, seconds_median=None, z_absdiff_max=None, nonfinite_z=None)
    save_json(path, result)
    return result


def best(results: list[dict]):
    """find the fastest passing case by median measured wall time."""
    return min((r for r in results if r["gate_passed"]), key=lambda r: r["seconds_median"], default=None)


def cases(args, results: Path, devices) -> list[tuple[str, int]]:
    """build a single-device sweep or replicate the single-device winner on all GPUs."""
    if args.variant:
        unknown = set(args.variant) - VARIANTS.keys()
        if unknown:
            raise ValueError(f"unknown variants {sorted(unknown)}; options: {sorted(VARIANTS)}")
        return [(name, args.chunk) for name in dict.fromkeys(args.variant)]
    if args.single_job:
        path = results / "json" / f"job-{args.single_job}" / "selection.json"
        record = json.loads(path.read_text())
        winner = record["winner"]
        if (winner is None or not winner["gate_passed"] or winner["gpu_count"] != 1
                or winner["mode"] != "full" or winner["slurm_job_id"] != args.single_job):
            raise ValueError(f"{path}: invalid single-GPU winner: {winner}")
        return list(dict.fromkeys([("baseline", args.chunk), (winner["variant"], winner["chunk"])]))
    return [(name, args.chunk) for name, spec in VARIANTS.items() if not spec[3] or devices[0].platform == "gpu"]


def run(args, results: Path) -> None:
    """run a reproducible sweep and leave a machine-readable selection record."""
    jax.config.update("jax_enable_x64", True)
    available = jax.devices("gpu") if args.gpu else jax.devices()
    if len(available) < args.devices:
        raise RuntimeError(f"required {args.devices} devices; found {available}")
    devices = available[:args.devices]
    job_id = os.environ.get("SLURM_JOB_ID")
    stamp = f"job-{job_id}" if job_id else f"{args.mode}-local-{time.time_ns()}"
    out = results / "json" / stamp
    data = load(args.mode)
    pending = cases(args, results, devices)
    logger.info("%s on %s: %s", stamp, devices, pending)
    records = [run_case(data, args.mode, name, chunk, devices, args.repeats, out, job_id) for name, chunk in pending]
    winner = best(records)
    if args.mode == "full" and args.devices == 1 and not args.variant and winner is not None:
        for chunk in (8192, 16384, 65536):
            if chunk != winner["chunk"]:
                records.append(run_case(data, args.mode, winner["variant"], chunk, devices, args.repeats, out, job_id))
        winner = best(records)
    save_json(out / "selection.json", {"winner": winner, "cases": [r["result_path"] for r in records]})
    bad_controls = [r["result_path"] for r in records if r["variant"] == "baseline" and not r["gate_passed"]]
    errors = [r["result_path"] for r in records if r["status"] == "error"]
    if bad_controls or errors or winner is None:
        raise RuntimeError(f"benchmark did not complete successfully under {out}: {bad_controls=}, {errors=}, {winner=}")
    logger.info("winner %s chunk=%s %.3f s; %s", winner["variant"], winner["chunk"], winner["seconds_median"], out)
