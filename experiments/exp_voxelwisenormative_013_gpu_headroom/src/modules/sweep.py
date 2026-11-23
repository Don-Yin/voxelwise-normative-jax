"""Fit all 64 exp_005 draws at each K refits per compiled call and compare every z with K=1, bit for bit."""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.batched_blr import N_ITER
from src.utils.receipts import peak_rss_bytes, publish, runtime

from .batched import Batch, Batcher
from .gpu_samples import Sampler
from .inputs import DRAW_SEED, Inputs, read_ixi, synthetic
from .trace import capture

logger = logging.getLogger(__name__)
K_VALUES = (1, 2, 4, 8, 16, 32, 64)
Z_TOLERANCE = 0.01  # GPU agreement with the converged PCNtoolkit reference, as exp_010
THRESHOLD = 1.96
MEMORY_KEYS = ("peak_bytes_in_use", "bytes_in_use", "peak_pool_bytes", "bytes_limit")


def open_device(smoke: bool):
    """A100 for real runs, CPU for the synthetic smoke; never fall back."""
    import jax

    jax.config.update("jax_enable_x64", True)
    device = jax.devices("cpu" if smoke else "gpu")[0]
    if not smoke and (device.platform != "gpu" or "A100" not in device.device_kind):
        raise RuntimeError(f"this sweep requires an NVIDIA A100, got {device.device_kind} on {device.platform}")
    return device, jax.__version__


def memory(device) -> dict | None:
    """Allocator counters; the peak is cumulative over the process, so passes run in ascending K."""
    stats = device.memory_stats()
    return None if stats is None else {key: int(stats[key]) for key in MEMORY_KEYS if key in stats}


def agreement(z: np.ndarray, ref: np.ndarray) -> tuple[bool, int, float, int]:
    """Bitwise equality, differing elements, max |Δz| and |z|>1.96 disagreements for one refit."""
    if np.array_equal(z, ref):
        return True, 0, 0.0, 0
    flips = np.count_nonzero((np.abs(z) > THRESHOLD) != (np.abs(ref) > THRESHOLD))
    return False, int(np.count_nonzero(z != ref)), float(np.max(np.abs(z.astype(np.float64) - ref))), int(flips)


def original_check(batcher: Batcher, inputs: Inputs) -> dict:
    """Fit every reference row once and check the pinned PCNtoolkit voxels before the sweep."""
    batch = batcher.run(np.asarray([len(inputs.rows)]))
    report = {"seconds_including_first_run": batch.compute_seconds + batch.gather_seconds,
              "flags": int(np.count_nonzero(np.abs(batch.z) > THRESHOLD)), "pinned_z_max_abs_delta": None}
    if inputs.pinned_voxels is not None:
        error = float(np.max(np.abs(batch.z[0][:, inputs.pinned_voxels] - inputs.pinned_z)))
        if not error <= Z_TOLERANCE:
            raise ValueError(f"original fit max |Δz| {error:g} against PCNtoolkit exceeds {Z_TOLERANCE}")
        report["pinned_z_max_abs_delta"] = error
    logger.info("original fit: %s", report)
    return report


def one_pass(batcher: Batcher, label: str, k: int, reference: np.ndarray, ref_steps: np.ndarray) -> dict:
    """Time every draw in batches of K; outside the timer, store or compare each refit's z."""
    if str(k) not in batcher.reports:
        batcher.compile(k)
    refits = len(reference)
    widths = np.asarray([hi - lo for lo, hi in batcher.bounds])
    record = {"label": label, "k": k, "status": "ok", "warm_seconds": batcher.warm(np.arange(k)), "batches": []}
    tally = {"bitwise_equal_refits": 0, "differing_elements": 0, "max_abs_dz": 0.0, "flag_disagreements": 0,
             "steps_equal_refits": 0}
    for b in range(refits // k):
        draws = np.arange(b * k, (b + 1) * k)
        batch: Batch = batcher.run(draws)
        if not np.isfinite(batch.z).all():
            raise ValueError(f"non-finite z for k={k} draws {draws.tolist()}: {np.count_nonzero(~np.isfinite(batch.z))}")
        for j, i in enumerate(draws):
            if label == "reference":
                reference[i], ref_steps[i] = batch.z[j], batch.steps[j]
                continue
            same, n, dz, flips = agreement(batch.z[j], reference[i])
            tally["bitwise_equal_refits"] += same
            tally["differing_elements"] += n
            tally["max_abs_dz"] = max(tally["max_abs_dz"], dz)
            tally["flag_disagreements"] += flips
            tally["steps_equal_refits"] += bool(np.array_equal(batch.steps[j], ref_steps[i]))
        record["batches"].append({
            "compute_seconds": batch.compute_seconds, "gather_seconds": batch.gather_seconds,
            "device_window": list(batch.device_window), "loop_steps": batch.steps.max(axis=0).tolist(),
            "computed_voxel_steps": int(k * (batch.steps.max(axis=0) * widths).sum()),
            "member_voxel_steps": int((batch.steps * widths).sum()), "active_voxel_steps": int(batch.voxel_steps.sum())})
        logger.info("[%d/%d] %s k=%d device %.3fs, gather %.3fs", b + 1, refits // k, label, k,
                    batch.compute_seconds, batch.gather_seconds)
    seconds = sum(x["compute_seconds"] + x["gather_seconds"] for x in record["batches"])
    record.update({"seconds_per_refit": seconds / refits, "memory_after": memory(batcher.device),
                   "agreement_with_k1": None if label == "reference" else tally})
    logger.info("%s k=%d: %.4fs per refit; agreement %s", label, k, seconds / refits, record["agreement_with_k1"])
    return record


def sweep(batcher: Batcher, k_values: tuple[int, ...]) -> list[dict]:
    """K=1 reference, K=1 repeat (determinism control), then ascending K until device memory runs out."""
    import jax

    refits, (nt, v) = len(batcher.inputs.rows), batcher.inputs.test.shape
    reference = np.empty((refits, nt, v), dtype=batcher.dtype)
    ref_steps = np.empty((refits, len(batcher.bounds)), dtype=np.int64)
    plan = [("reference", 1), ("repeat", 1), *(("batched", k) for k in k_values if k > 1 and refits % k == 0)]
    passes = []
    for label, k in plan:
        try:
            passes.append(one_pass(batcher, label, k, reference, ref_steps))
        except jax.errors.JaxRuntimeError as error:
            if "RESOURCE_EXHAUSTED" not in str(error) or label != "batched":
                raise
            passes.append({"label": label, "k": k, "status": "out_of_memory", "error": str(error)[:2000],
                           "memory_after": memory(batcher.device)})
            logger.warning("k=%d ran out of device memory; stopping the sweep: %s", k, str(error).splitlines()[0])
            break
    return passes


def run(results: Path, run_name: str, precision: str, chunk: int, *, smoke: bool) -> Path:
    """One precision in one process: sweep, publish the receipt, then trace K=1 and the largest K that fitted."""
    out = results / "json" / run_name / f"{precision}.json"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    logger.info("start sweep run=%s precision=%s smoke=%s chunk=%d output=%s", run_name, precision, smoke, chunk, out)
    device, version = open_device(smoke)
    start = time.perf_counter()
    inputs = synthetic() if smoke else read_ixi()
    input_seconds = time.perf_counter() - start
    batcher = Batcher(inputs, chunk, precision, device)
    batcher.compile(1)
    original = original_check(batcher, inputs)
    sampler = None if smoke else Sampler(results / "samples" / run_name / f"nvidia-smi-{precision}.csv")
    with sampler or contextlib.nullcontext():
        passes = sweep(batcher, K_VALUES)
    n, v = inputs.train.shape
    publish(out, {
        "schema_version": 1, "run_name": run_name, "precision": precision, "mode": "synthetic-cpu" if smoke else "ixi-cuda",
        "device": device.device_kind, "jax": version, "chunk": chunk, "chunk_widths": [hi - lo for lo, hi in batcher.bounds],
        "references": n, "held_out": len(inputs.test), "voxels": v, "refits": len(inputs.rows), "draw_seed": DRAW_SEED,
        "draw_digest": inputs.draw_digest, "design_seconds_all_draws": inputs.design_seconds, "input_seconds": input_seconds,
        "kernel": "early_newton.make_variant(analytic=True, early=True, tf32=False), vmapped over refits",
        "max_steps": N_ITER, "z_threshold": THRESHOLD, "original_fit": original, "compile": batcher.reports,
        "passes": passes,
        "gpu_sampler": None if sampler is None else {"path": str(sampler.path.relative_to(results)), **sampler.report},
        "host_peak_rss_bytes": peak_rss_bytes(), **runtime()})
    fitted = [p["k"] for p in passes if p["status"] == "ok"]
    traces = [capture(batcher, k, results, Path("profile", "jax-trace", run_name, f"{precision}-k{k}"))
              for k in sorted({1, max(fitted)})]
    publish(out.with_name(f"{precision}-trace.json"), {"schema_version": 1, "run_name": run_name,
                                                        "precision": precision, "traces": traces})
    return out
