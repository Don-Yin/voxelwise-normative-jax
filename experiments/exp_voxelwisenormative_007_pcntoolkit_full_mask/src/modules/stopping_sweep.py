"""Separate the function-tolerance stop from the line-search limit on the ten largest-gap voxels."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from shared.voxelwisenormative.cohort import standardise

from .gradient_benchmark import load_selected, package_versions, read_audit
from .gradient_fit import compile_value_and_grad, timed_fit, verify_gradient

logger = logging.getLogger(__name__)
FACTR = (1e7, 1e3, 10.0)  # 1e7 is SciPy's default and so PCNtoolkit's.
MAXLS = (20, 50)  # 20 is SciPy's default and so PCNtoolkit's.
Z_TOLERANCE = 1e-3  # The project's z agreement target.
ORIGINAL = "original-finite-difference"


def cell_record(trials, column, z, gpu_z, loss, gpu_loss, archive_z=None):
    """Condense every attempt of one fit; keep the last attempt's stop and iteration trace."""
    last = trials[-1]
    delta = np.abs(z - gpu_z)
    gradients = [t["gradient_calls"] for t in trials]
    trace = last["objective_trace"]
    record = {
        "mask_column": column, "attempts": len(trials), "solver": last["solver"],
        "ftol": last["solver"]["factr"] * np.finfo(float).eps, "task": last["task"], "warnflag": last["warnflag"],
        "iterations": last["iterations"], "gradient_inf": last["gradient_inf"],
        "function_calls": sum(t["function_calls"] for t in trials),
        "gradient_calls": None if None in gradients else sum(gradients),
        "sentinel_evaluations": sum(t["sentinel_evaluations"] for t in trials),
        "optimizer_seconds": sum(t["optimizer_seconds"] for t in trials),
        "penalised_nll": float(loss), "gpu_penalised_nll": float(gpu_loss),
        "objective_gap_to_gpu": float(loss - gpu_loss), "max_abs_delta_z_vs_gpu": float(delta.max()),
        "worst_gpu_test_row": int(delta.argmax()), "objective_trace": trace, "step_trace": last["step_trace"],
        "last_relative_reduction": ((trace[-2] - trace[-1]) / max(abs(trace[-2]), abs(trace[-1]), 1.0)
                                    if len(trace) > 1 else None),
        "last_step_norm": last["step_trace"][-1] if last["step_trace"] else None,
    }
    if archive_z is not None:
        record["cpu_archive_max_abs_delta_z"] = float(np.abs(z - archive_z).max())
    return record


def summarise(records):
    """Report agreement, stops and effort for one condition without hiding failed fits."""
    done = [r for r in records if "error" not in r]
    summary = {"n_voxels": len(records), "failed": len(records) - len(done)}
    if done:
        iterations = [r["iterations"] for r in done]
        summary |= {
            "n_within_z_tolerance": sum(r["max_abs_delta_z_vs_gpu"] <= Z_TOLERANCE for r in done),
            "max_abs_delta_z_vs_gpu": max(r["max_abs_delta_z_vs_gpu"] for r in done),
            "stop_counts": dict(Counter(r["task"] for r in done)),
            "iterations_min_median_max": [min(iterations), float(np.median(iterations)), max(iterations)],
            "function_calls_total": sum(r["function_calls"] for r in done),
            "sentinel_evaluations_total": sum(r["sentinel_evaluations"] for r in done),
            "optimizer_seconds_total": sum(r["optimizer_seconds"] for r in done),
        }
    return summary


def sweep(results: Path, array: str, merge: str, calibration: str, audit: str) -> None:
    """Fit each audited voxel under the original stop settings and then looser and tighter ones."""
    versions = package_versions()
    worst = read_audit(audit, (array, merge, calibration))["worst_voxels"]
    cols = np.array([v["mask_column"] for v in worst], dtype=int)
    gpu_loss = [v["gpu_from_archive"]["penalised_nll"] for v in worst]
    gm, cov, ref, cpu, _, gpu = load_selected(results, array, merge, calibration, cols)
    y, yt = standardise(gm, ref["test"])
    phi, phit = ref["Phi_train"], ref["Phi_test"]
    start = perf_counter()
    bind = compile_value_and_grad(phi, y[:, 0])
    warmup_seconds = perf_counter() - start
    valid = verify_gradient(bind, phi, phit, y, yt, range(len(cols)))
    logger.info("CPU JAX warm-up %.2f s; objective and gradient checks %s", warmup_seconds, valid)

    out = results / "stopping-sweep" / os.environ.get("SLURM_JOB_ID", "local")
    out.mkdir(parents=True, exist_ok=False)
    conditions = [(ORIGINAL, {}, None)] + [(f"jax-factr-{f:g}-maxls-{m}", {"factr": f, "maxls": m}, bind)
                                           for f, m in product(FACTR, MAXLS)]
    records: dict[str, list[dict]] = {}
    for name, solver, gradient in conditions:
        records[name] = []
        for j, column in enumerate(cols.tolist()):
            # One voxel per call: a failure in one fit cannot discard the rest of the sweep.
            try:
                z, loss, fitted = timed_fit(gm, cov, ref, np.array([j]), out / name, gradient, solver)
                records[name].append(cell_record(fitted[0]["attempts"], column, z[:, 0], gpu[:, j], loss[0], gpu_loss[j],
                                                 cpu[:, j] if name == ORIGINAL else None))
            except Exception as exc:  # record the failed fit and continue with the other conditions
                logger.exception("%s failed at mask column %d", name, column)
                records[name].append({"mask_column": column, "error": f"{type(exc).__name__}: {exc}"})
        logger.info("%s: %s", name, json.dumps(summarise(records[name])))

    summaries = {name: summarise(rec) for name, rec in records.items()}
    receipt = {"job": out.name, "versions": versions, "array_job": array, "merge_job": merge,
               "calibration_job": calibration, "audit_job": audit, "columns": cols.tolist(),
               "z_tolerance": Z_TOLERANCE, "jax_warmup_and_compile_seconds": warmup_seconds,
               "objective_and_gradient_check": valid, "summary": summaries, "per_voxel": records}
    output = out / "sweep.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temp.rename(output)
    logger.info("saved %s", output)
    if any(s["failed"] for s in summaries.values()):
        raise RuntimeError(f"failed fits in {[n for n, s in summaries.items() if s['failed']]}; inspect {output}")
