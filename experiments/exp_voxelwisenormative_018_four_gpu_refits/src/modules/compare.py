"""Warm 1-vs-4-device throughput, whole-map agreement and a warm single-chunk trace."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import jax
import numpy as np

from shared.voxelwisenormative.batch_job import require_hugging_face_wrapper
from shared.voxelwisenormative.receipts import publish
from shared.voxelwisenormative.refit_batch import (
    THRESHOLD,
    Z_TOLERANCE,
    Batcher,
    agreement,
    original_check,
)
from shared.voxelwisenormative.refit_inputs import (
    DRAW_SEED,
    Inputs,
    read_ixi,
    synthetic,
)
from shared.voxelwisenormative.refit_trace import capture, device_busy

logger = logging.getLogger(__name__)
N_DEVICES, CHUNK, DTYPE = 4, 32_768, "float64"
SAME_KERNEL_Z_TOLERANCE = 1e-8


def prepare(inputs: Inputs, chunk: int, device) -> tuple[Batcher, dict]:
    """Upload inputs and compile/warm both widths outside the refit timer."""
    start = time.perf_counter()
    batcher = Batcher(inputs, chunk, DTYPE, device)
    jax.block_until_ready((batcher.train, batcher.test))
    upload = time.perf_counter() - start
    batcher.compile(1)
    warm = batcher.warm(np.asarray([0]))
    return batcher, {"device_id": device.id, "upload_seconds": upload, "warm_seconds": warm,
                     "compile_by_chunk_width": batcher.reports["1"], "wall_seconds": time.perf_counter() - start}


def shard(batcher: Batcher, first: int, stop: int, z: np.ndarray, barrier: Barrier) -> dict:
    """Run one contiguous shard serially on its device and gather into disjoint host slots."""
    barrier.wait()
    start = time.perf_counter()
    device_s, gather_s = 0.0, 0.0
    for i in range(first, stop):
        batch = batcher.run(np.asarray([i]), out_z=z[i:i + 1])
        device_s += batch.compute_seconds
        gather_s += batch.gather_seconds
        logger.info("[%d/%d] device %s draw %s: ready %.3fs, gather %.3fs", i - first + 1, stop - first,
                    batcher.device.id, i, batch.compute_seconds, batch.gather_seconds)
    return {"device_id": batcher.device.id, "start_draw": first, "stop_draw": stop, "refits": stop - first,
            "wall_seconds": time.perf_counter() - start, "sum_dispatch_wait_seconds": device_s,
            "sum_gather_seconds": gather_s}


def refit_pass(batchers: list[Batcher]) -> tuple[np.ndarray, dict]:
    """Time the full warm pass from thread start until the last z map has landed on the host."""
    inputs = batchers[0].inputs
    n = len(inputs.rows)
    size, extra = divmod(n, len(batchers))
    if extra:
        raise ValueError(f"cannot split {n} refits evenly across {len(batchers)} devices")
    z = np.empty((n, *inputs.test.shape), dtype=DTYPE)
    barrier = Barrier(len(batchers))
    assignments = [(batcher, i * size, (i + 1) * size, z, barrier) for i, batcher in enumerate(batchers)]
    logger.info("start timed pass: %d devices, %d refits per device", len(batchers), size)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, len(batchers) - 1)) as pool:
        futures = [pool.submit(shard, *a) for a in assignments[1:]]
        shards = [shard(*assignments[0]), *(future.result() for future in futures)]
    seconds = time.perf_counter() - start
    logger.info("timed pass finished: %d devices, %d refits, %.3fs", len(batchers), n, seconds)
    return z, {"wall_seconds": seconds, "refits_per_second": n / seconds, "shards": shards}


def compare_z(single: np.ndarray, four: np.ndarray) -> dict:
    """Check every finite value in matching draw order; report numerical and call agreement."""
    if single.shape != four.shape:
        raise ValueError(f"one-device z shape {single.shape} differs from four-device shape {four.shape}")
    report = {"compared_scores": single.size, "bitwise_equal_refits": 0, "differing_elements": 0,
              "max_abs_dz": 0.0, "flag_disagreements": 0}
    for draw, (ref, candidate) in enumerate(zip(single, four, strict=True)):
        bad_ref, bad_candidate = (int(np.count_nonzero(~np.isfinite(z))) for z in (ref, candidate))
        if bad_ref or bad_candidate:
            raise ValueError(f"non-finite z at draw {draw}: one device {bad_ref}, four devices {bad_candidate}")
        equal, different, delta, flips = agreement(candidate, ref)
        report["bitwise_equal_refits"] += equal
        report["differing_elements"] += different
        report["max_abs_dz"] = max(report["max_abs_dz"], delta)
        report["flag_disagreements"] += flips
    return {**report, "z_threshold": THRESHOLD, "max_abs_dz_allowed": SAME_KERNEL_Z_TOLERANCE,
            "passed": report["max_abs_dz"] <= SAME_KERNEL_Z_TOLERANCE and report["flag_disagreements"] == 0}


def run(results: Path, run_name: str, *, smoke: bool) -> None:
    """Run the one-device baseline, the four-device pass, and one trace outside timing."""
    job = None if smoke else require_hugging_face_wrapper()
    jax.config.update("jax_enable_x64", True)
    backend = "cpu" if smoke else "gpu"
    devices = jax.devices(backend)
    if len(devices) != N_DEVICES or (not smoke and not all("A100" in d.device_kind for d in devices)):
        raise RuntimeError(f"expected {N_DEVICES} {'logical CPU devices' if smoke else 'NVIDIA A100s'}, got {devices}")
    logger.info("start run %s on devices %s", run_name, devices)
    out = results / "json" / run_name
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    inputs = synthetic() if smoke else read_ixi()
    input_seconds = time.perf_counter() - start
    chunk = 64 if smoke else CHUNK
    first, first_prep = prepare(inputs, chunk, devices[0])
    original = original_check(first, inputs)
    one_z, one = refit_pass([first])
    others = [prepare(inputs, chunk, device) for device in devices[1:]]
    four_z, four = refit_pass([first, *(batcher for batcher, _ in others)])
    compare_start = time.perf_counter()
    validation = compare_z(one_z, four_z)
    comparison_seconds = time.perf_counter() - compare_start
    del one_z, four_z
    speedup = one["wall_seconds"] / four["wall_seconds"]
    publish(out / "refits.json", {
        "schema_version": 1, "run_name": run_name, "mode": "synthetic-cpu" if smoke else "ixi-cuda",
        "hugging_face_job": job, "jax_version": jax.__version__, "precision": DTYPE,
        "devices": [{"id": d.id, "kind": d.device_kind, "platform": d.platform} for d in devices],
        "references": inputs.train.shape[0], "held_out": inputs.test.shape[0], "voxels": inputs.test.shape[1],
        "refits": len(inputs.rows), "draw_seed": DRAW_SEED, "draw_digest": inputs.draw_digest,
        "chunk_widths": [hi - lo for lo, hi in first.bounds], "input_seconds": input_seconds,
        "design_seconds": inputs.design_seconds, "original_fit": {**original, "pinned_z_max_abs_delta_allowed": Z_TOLERANCE},
        "preparation": [first_prep, *(record for _, record in others)], "one_device": one, "four_devices": four,
        "speedup": speedup, "parallel_efficiency": speedup / N_DEVICES, "agreement": validation,
        "comparison_seconds": comparison_seconds,
    })
    trace = capture(first, 1, results, Path("profile", "jax-trace", run_name, "first-chunk"))
    activity = device_busy(results / trace["perfetto"], include_kernels=True)
    publish(out / "trace.json", {"device_id": first.device.id, "draw_index": 0, **trace, **activity})
    if not smoke and not activity.get("top_kernels"):
        raise RuntimeError(f"no GPU kernels in {results / trace['perfetto']}")
    if not validation["passed"]:
        raise ValueError(f"one- vs four-device z agreement failed: {validation}")
