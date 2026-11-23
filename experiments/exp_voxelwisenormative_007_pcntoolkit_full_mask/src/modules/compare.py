"""Join the full CPU array and compare it with the frozen CUDA fit."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.artifacts import PCNTOOLKIT_REFERENCE, resolve
from shared.voxelwisenormative.cohort import standardise

from .campaign import check_job, inputs, read_calibration, reference, shard_bounds
from .fit import DESIGN_ATOL, DESIGN_COLS, SHAPE

logger = logging.getLogger(__name__)
CUDA = "batched-blr-full-cuda-165402-v1"


def read_shards(results: Path, calibration: str, array: str, shards: int, test):
    """Reject missing, duplicate, nonfinite, unbound or misordered shard payloads."""
    directory = results / "arrays" / check_job(array)
    expected = {directory / f"shard-{i:03d}.npz" for i in range(shards)}
    actual = set(directory.glob("shard-*.npz"))
    if actual != expected:
        raise RuntimeError(f"array {array} shard set differs: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    z = np.empty((int(test.sum()), SHAPE[1]), dtype=np.float64)
    w = np.empty((SHAPE[1], DESIGN_COLS), dtype=np.float64)
    nlz = np.empty(SHAPE[1], dtype=np.float64)
    for i in range(shards):
        lo, hi = shard_bounds(i, shards)
        path = directory / f"shard-{i:03d}.npz"
        with np.load(path, allow_pickle=False) as f:
            if (f["array"].item(), f["calibration"].item(), f["shards"].item()) != (array, calibration, shards):
                raise ValueError(f"shard receipt differs: {path}")
            np.testing.assert_array_equal(f["test"], test, err_msg=f"split: {path}")
            np.testing.assert_array_equal(f["voxels"], np.arange(lo, hi), err_msg=f"columns: {path}")
            for name, dest in (("z_test", z[:, lo:hi]), ("posterior_weights", w[lo:hi]), ("nlZ", nlz[lo:hi])):
                a = f[name]
                if a.shape != dest.shape or a.dtype != np.float64 or not np.isfinite(a).all():
                    raise ValueError(f"{path}::{name} expected finite float64 {dest.shape}, got {a.dtype} {a.shape}")
                dest[:] = a
    return z, w, nlz


def sex_column(cov, ref: dict) -> int:
    """Identify the single standardised sex column in both original designs."""
    sex = (cov["sex"] == "M").astype(float)
    test = ref["test"]
    scaled = (sex - sex[~test].mean()) / sex[~test].std()
    design = np.concatenate((ref["Phi_train"], ref["Phi_test"]))
    encoded = np.concatenate((scaled[~test], scaled[test]))
    match = np.flatnonzero(np.all(np.isclose(design, encoded[:, None], rtol=0, atol=DESIGN_ATOL), axis=0))
    if match.shape != (1,):
        raise ValueError(f"expected one sex design column, got {match.tolist()}")
    return int(match[0])


def recovered_weights(gm, gpu_z, hyp, ref: dict):
    """Recover posterior means at archived hyperparameters and check the row-0 prediction."""
    phi, phit = ref["Phi_train"], ref["Phi_test"]
    lam, Q = np.linalg.eigh(phi.T @ phi)
    lam = lam.clip(0)
    G, H0 = (phi @ Q).T, phit[0] @ Q
    w = np.empty((SHAPE[1], DESIGN_COLS), dtype=np.float64)
    max_error = 0.0
    for lo in range(0, SHAPE[1], 4096):
        hi = min(lo + 4096, SHAPE[1])
        y, yt = standardise(gm[:, lo:hi], ref["test"])
        logbeta, eps, logb, logalpha = hyp[lo:hi].astype(np.float64).T
        beta, b, alpha = np.exp(logbeta), np.exp(logb), np.exp(logalpha)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            den = lam[:, None] * beta[None, :] + alpha[None, :]
            rz = G @ np.sinh(b * (np.arcsinh(y) + eps))
            mean = beta * rz / den
            w[lo:hi] = (Q @ mean).T
            var0 = 1.0 / beta + (H0**2) @ (1.0 / den)
            pred = (np.sinh(b * (np.arcsinh(yt[0]) + eps)) - H0 @ mean) / np.sqrt(var0)
        max_error = max(max_error, float(np.max(np.abs(pred - gpu_z[0, lo:hi]))))
    # Stored float32 hyperparameters and scores were rounded independently.
    if not np.isfinite(w).all() or max_error > 1e-4:
        raise ValueError(f"CUDA posterior recovery does not reproduce archived row 0: max |Δz| {max_error}")
    return w, max_error


def agreement(first, second) -> dict:
    """Summarise differences without dropping even the worst mask column."""
    delta = np.abs(second - first)
    if not np.isfinite(delta).all():
        raise ValueError("nonfinite differences")
    idx = np.unravel_index(delta.argmax(), delta.shape)
    return {"n": delta.size, "max_abs": float(delta[idx]), "p999_abs": float(np.quantile(delta, 0.999)),
            "worst_index": [int(x) for x in idx]}


def merge(results: Path, calibration: str, array: str, shards: int) -> None:
    """Publish complete HPC arrays, then a small image input with no grey-matter intensities."""
    array = check_job(array)
    out = results / "merged" / array
    if out.exists():
        raise FileExistsError(out)
    verified = read_calibration(results, calibration)
    ref = reference(PCNTOOLKIT_REFERENCE, "voxels", "y_train")
    gm, cov, _ = inputs(ref, verified)
    column = sex_column(cov, ref)
    paths = resolve(CUDA, "z_test.npy", "hyp.npy", "result.json")
    provenance = json.loads(paths["result.json"].read_text())
    gpu_z = np.load(paths["z_test.npy"], mmap_mode="r", allow_pickle=False)
    hyp = np.load(paths["hyp.npy"], mmap_mode="r", allow_pickle=False)
    if (gpu_z.shape, hyp.shape, provenance["device"]) != ((113, SHAPE[1]), (SHAPE[1], 4), "cuda:0"):
        raise ValueError(f"wrong CUDA sources: {paths}, {gpu_z.shape=} {hyp.shape=} {provenance['device']=}")
    np.testing.assert_allclose(standardise(gm[:, ref["voxels"]], ref["test"])[0], ref["y_train"],
                               rtol=0, atol=1e-9, err_msg="CUDA training response standardisation")
    z, w, nlz = read_shards(results, calibration, array, shards, ref["test"])
    gpu_w, recovery_error = recovered_weights(gm, gpu_z, hyp, ref)
    stats = {
        "calibration_job": calibration, "array_job": array, "merge_job": os.environ.get("SLURM_JOB_ID"),
        "shards": shards, "n_voxels": SHAPE[1], "n_test": 113, "test_row": 0,
        "cohort_row": int(np.flatnonzero(ref["test"])[0]), "sex_design_column_zero_based": column,
        "z_all": agreement(z, gpu_z), "z_row_0": agreement(z[0], gpu_z[0]),
        "sex_weight": agreement(w[:, column], gpu_w[:, column]), "gpu_recovery_max_abs_delta_z": recovery_error,
        "gpu": "exp_003 float64 CUDA fit; z and hyperparameters archived as float32",
    }
    stage = out.with_name(f".{array}-{os.getpid()}.tmp")
    stage.mkdir(parents=True)
    for name, a in (("z_test", z), ("posterior_weights", w), ("nlZ", nlz), ("gpu_posterior_weights", gpu_w)):
        np.save(stage / f"{name}.npy", a, allow_pickle=False)
    np.savez_compressed(stage / "display.npz", pcn_z=z[0], gpu_z=gpu_z[0], pcn_sex=w[:, column], gpu_sex=gpu_w[:, column])
    (stage / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    stage.rename(out)
    logger.info("saved complete comparison %s: %s", out, json.dumps(stats))
