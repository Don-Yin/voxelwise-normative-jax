"""Measure both gradient choices on the calibration and largest-gap voxels."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from shared.voxelwisenormative.artifacts import PCNTOOLKIT_REFERENCE, resolve
from shared.voxelwisenormative.cohort import standardise

from .audit_outliers import read_archive
from .campaign import CALIB_VOXELS, check_job, inputs, read_calibration, reference
from .fit import EPSILON, SHAPE
from .gradient_fit import compile_value_and_grad, timed_fit, verify_gradient

logger = logging.getLogger(__name__)
MODES = ("finite_difference", "jax_gradient")
BLOCK_SIZE = 64  # Limit model/scaler state retained alongside the JAX runtime.


def package_versions() -> dict[str, str]:
    """Require the installed runtime that produced the archived CPU fit."""
    versions = {name: version(name) for name in ("pcntoolkit", "scipy", "jax", "jaxlib")}
    if (versions["pcntoolkit"], versions["scipy"]) != ("1.4.0", "1.18.1"):
        raise RuntimeError(f"expected PCNtoolkit 1.4.0 / SciPy 1.18.1, got {versions}")
    return versions


def read_audit(audit: str, jobs: tuple[str, str, str]) -> dict:
    """Load the hash-verified audit after checking that it describes the bound jobs."""
    path = resolve("exp007-outlier-display-v1", "outliers.json")["outliers.json"]
    try:
        audit_data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot read audit payload {path}") from exc
    if tuple(audit_data[key] for key in ("array_job", "merge_job", "calibration_job", "job")) != (*jobs, check_job(audit)):
        raise ValueError(f"wrong source jobs for {path}; expected {(*jobs, audit)}")
    worst = np.array([x["mask_column"] for x in audit_data["worst_voxels"]], dtype=int)
    if worst.size != 10 or np.unique(worst).size != 10 or np.any((worst < 0) | (worst >= SHAPE[1])):
        raise ValueError(f"expected 10 distinct valid audited worst voxels: {worst.tolist()}")
    return audit_data


def selection(audit: str, jobs: tuple[str, str, str]):
    """Use the hash-verified top ten and the exact 512 calibration columns."""
    worst = np.array([x["mask_column"] for x in read_audit(audit, jobs)["worst_voxels"]], dtype=int)
    calibration = np.linspace(0, SHAPE[1] - 1, CALIB_VOXELS, dtype=int)
    return np.union1d(calibration, worst), set(calibration.tolist()), set(worst.tolist())


def load_selected(results, array, merge, calibration, cols):
    """Only keep the 522 selected columns in RAM after verifying archive identity."""
    cpu, gpu, _ = read_archive(results, array, merge, calibration)
    ref = reference(PCNTOOLKIT_REFERENCE)
    gm, cov, _ = inputs(ref, read_calibration(results, calibration))
    return (np.ascontiguousarray(gm[:, cols]), cov, ref, cpu["z_test.npy"][:, cols],
            cpu["nlZ.npy"][cols], gpu["z_test.npy"][:, cols])


def annotate(record, col, groups, z, gpu, objective):
    """Add all attempts' evaluation counts and the largest held-out score difference."""
    trials = record["attempts"]
    gradients = [r["gradient_calls"] for r in trials]
    if not trials or not np.isfinite(gpu).all():
        raise ValueError(f"missing fit attempts or nonfinite GPU scores at mask column {col}")
    delta = np.abs(z - gpu)
    record.update(mask_column=col, calibration=col in groups[0], audited_outlier=col in groups[1],
                  penalised_nll=float(objective), max_abs_delta_z_vs_gpu=float(delta.max()),
                  worst_gpu_test_row=int(delta.argmax()), optimizer_seconds=sum(r["optimizer_seconds"] for r in trials),
                  function_calls=sum(r["function_calls"] for r in trials),
                  gradient_calls=None if None in gradients else sum(gradients))


def summarise(records):
    """Summarise speed, score agreement and solver exits without dropping outliers."""
    return {
        "n_voxels": len(records),
        "max_abs_delta_z_vs_gpu": max(x["max_abs_delta_z_vs_gpu"] for x in records),
        "blr_fit_seconds_total": sum(x["blr_fit_seconds"] for x in records),
        "blr_fit_seconds_median": float(np.median([x["blr_fit_seconds"] for x in records])),
        "optimizer_seconds_total": sum(x["optimizer_seconds"] for x in records),
        "function_calls_total": sum(x["function_calls"] for x in records),
        "gradient_calls_total": (sum(x["gradient_calls"] for x in records)
                                 if all(x["gradient_calls"] is not None for x in records) else None),
        "stop_counts": dict(Counter(x["attempts"][-1]["task"] for x in records)),
    }


def benchmark(results: Path, array: str, merge: str, calibration: str, audit: str) -> None:
    """Repeat the original fit and the JAX fprime fit once on the same voxel set."""
    versions = package_versions()
    cols, *groups = selection(audit, (array, merge, calibration))
    gm, cov, ref, cpu, cpu_obj, gpu = load_selected(results, array, merge, calibration, cols)
    y, yt = standardise(gm, ref["test"])
    phi, phit = ref["Phi_train"], ref["Phi_test"]
    start = perf_counter()
    bind = compile_value_and_grad(phi, y[:, 0])
    warmup_seconds = perf_counter() - start
    check_ids = sorted({0, len(cols) // 2} | set(np.flatnonzero(np.isin(cols, list(groups[1])))))
    valid = verify_gradient(bind, phi, phit, y, yt, check_ids)
    logger.info("CPU JAX warm-up %.2f s; objective and gradient checks %s", warmup_seconds, valid)

    job = os.environ.get("SLURM_JOB_ID", "local")
    out = results / "gradient-benchmark" / job
    out.mkdir(parents=True, exist_ok=False)
    records: dict[str, list[dict]] = {mode: [] for mode in MODES}
    for batch, lo in enumerate(range(0, len(cols), BLOCK_SIZE)):
        hi = min(lo + BLOCK_SIZE, len(cols))
        ids = np.arange(lo, hi)
        # Alternate first arm between blocks; both see the same grouping and rows.
        for mode in MODES if batch % 2 == 0 else MODES[::-1]:
            z, loss, recs = timed_fit(gm, cov, ref, ids, out / f"{mode}-{batch:02d}",
                                     bind if mode == "jax_gradient" else None)
            for j, i in enumerate(ids):
                r = recs[j]
                annotate(r, int(cols[i]), groups, z[:, j], gpu[:, i], loss[j])
                if mode == "finite_difference":
                    r["cpu_archive_max_abs_delta_z"] = float(np.abs(z[:, j] - cpu[:, i]).max())
                    r["cpu_archive_abs_delta_nll"] = abs(float(loss[j] - cpu_obj[i]))
            records[mode].extend(recs)
        logger.info("[%d/%d] fitted both gradient choices", hi, len(cols))

    summaries = {mode: {"all": summarise(rec), "calibration_512": summarise([r for r in rec if r["calibration"]]),
                        "audited_10": summarise([r for r in rec if r["audited_outlier"]])}
                 for mode, rec in records.items()}
    drift = [r["mask_column"] for r in records["finite_difference"]
             if r["cpu_archive_max_abs_delta_z"] > 1e-3 or r["cpu_archive_abs_delta_nll"] > 1e-5]
    receipt = {"job": job, "versions": versions, "array_job": array, "merge_job": merge, "calibration_job": calibration,
               "audit_job": audit, "columns": cols.tolist(), "pcntoolkit_epsilon": EPSILON,
               "jax_warmup_and_compile_seconds": warmup_seconds, "objective_and_gradient_check": valid,
               "finite_difference_archive_drift_columns": drift, "summary": summaries, "per_voxel": records}
    output = out / "benchmark.json"
    temp = output.with_suffix(".tmp")
    temp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    temp.rename(output)
    logger.info("saved %s: %s", output, json.dumps(summaries))
    if drift:
        logger.warning("finite-difference replay differed from the full-mask archive in %d voxels: %s", len(drift), drift)
