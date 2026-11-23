"""Calibrate and fit the Slurm CPU array with a pinned matrix and split."""

from __future__ import annotations

import json
import logging
import os
import re
from importlib import import_module
from pathlib import Path
from time import perf_counter

import numpy as np

from shared.voxelwisenormative.artifacts import (
    ARTIFACTS,
    PCNTOOLKIT_REFERENCE,
    VOXEL_MATRIX,
    resolve,
    sha256,
)
from shared.voxelwisenormative.cohort import load_covariates, site_split

from .fit import DESIGN_COLS, EPSILON, SHAPE, fit_groups

logger = logging.getLogger(__name__)
CALIB_REF = "pcntoolkit-blr-reference-n20-v1"
CALIB_Z_ATOL = 1e-3
CALIB_OBJECTIVE_RTOL = 1e-8
CALIB_VOXELS = 512


def reference(artifact: str, *keys: str):
    """Load verified design, split and only the requested archived quantities."""
    path = resolve(artifact, "arrays.npz")["arrays.npz"]
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in ("test", "Phi_train", "Phi_test", *keys)}


def matrix_identity(path: Path) -> dict:
    """Cheap identity check for the frozen matrix after a verified read."""
    stat = path.stat()
    return {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}


def inputs(ref: dict, calibrated: dict | None = None):
    """Verify the split and matrix; only the calibration has to rehash all raw bytes."""
    cov = load_covariates()
    test = site_split(cov["site"], 0)
    if test.shape != (SHAPE[0],) or test.sum() != 113 or not np.array_equal(test, ref["test"]):
        raise ValueError(f"seed-0 split mismatch: {test.shape}, {int(test.sum())} test rows")
    for key, n in (("Phi_train", 450), ("Phi_test", 113)):
        if ref[key].shape != (n, DESIGN_COLS) or not np.isfinite(ref[key]).all():
            raise ValueError(f"{key}: expected finite {(n, DESIGN_COLS)}, got {ref[key].shape}")
    path = (ARTIFACTS / VOXEL_MATRIX / "gm.npy").resolve(strict=True)
    matrix = matrix_identity(path)
    if calibrated is None:
        resolve(VOXEL_MATRIX, "gm.npy")
        if matrix != matrix_identity(path):
            raise RuntimeError(f"matrix changed during verification: {path}")
    elif matrix != calibrated["matrix"]:
        raise RuntimeError(f"matrix does not match calibration: actual={matrix}, expected={calibrated['matrix']}")
    gm = np.load(path, mmap_mode="r", allow_pickle=False)
    if gm.shape != SHAPE or gm.dtype != np.float32:
        raise ValueError(f"{path}: expected float32 {SHAPE}, got {gm.dtype} {gm.shape}")
    return gm, cov, matrix


def write_new(path: Path, **arrays) -> None:
    """Publish a complete array file atomically without replacing a prior product."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("xb") as stream:
            np.savez(stream, **arrays)
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def calibrate(results: Path, parity_job: str | None) -> None:
    """Compare 20 voxels with exp_002; only then measure the real 512-voxel stage."""
    job = os.environ.get("SLURM_JOB_ID", "local")
    out = results / "calibration" / job
    if out.exists():
        raise FileExistsError(out)
    ref = reference(CALIB_REF, "voxels", "z_test", "nlZ")
    gm, cov, matrix = inputs(ref)
    voxels = ref["voxels"]
    np.testing.assert_array_equal(voxels, np.sort(np.random.default_rng(0).choice(SHAPE[1], 20, replace=False)))
    if parity_job is None:
        z, weights, nlz = fit_groups(gm, cov, ref, voxels, out / "pcntoolkit")
        write_new(out / "pairs.npz", voxels=voxels, z_test=z, nlZ=nlz, posterior_weights=weights)
    else:
        path = results / "calibration" / check_job(parity_job) / "pairs.npz"
        with np.load(path, allow_pickle=False) as pairs:
            np.testing.assert_array_equal(pairs["voxels"], voxels, err_msg=str(path))
            z, nlz = pairs["z_test"], pairs["nlZ"]
    if z.shape != (113, 20) or nlz.shape != (20,) or not (np.isfinite(z).all() and np.isfinite(nlz).all()):
        raise ValueError(f"invalid calibration arrays from job {parity_job or job}: {z.shape=} {nlz.shape=}")
    dz = np.abs(z - ref["z_test"])
    dnlz = np.abs(nlz - ref["nlZ"]) / np.maximum(np.abs(ref["nlZ"]), 1)
    i, j = np.unravel_index(dz.argmax(), dz.shape)
    receipt = {
        "job": job, "parity_job": parity_job or job, "matrix": matrix, "pcntoolkit": "1.4.0", "epsilon": EPSILON,
        "n_test": 113, "n_train": 450, "n_check_voxels": len(voxels),
        "max_abs_delta_z": float(dz[i, j]), "worst_test_row": int(i), "worst_mask_column": int(voxels[j]),
        "max_relative_delta_objective": float(dnlz.max()), "z_tolerance": CALIB_Z_ATOL,
        "objective_relative_tolerance": CALIB_OBJECTIVE_RTOL,
    }
    logger.info("calibration comparison %s", json.dumps(receipt))
    if dz.max() > CALIB_Z_ATOL or dnlz.max() > CALIB_OBJECTIVE_RTOL:
        raise RuntimeError(f"20-voxel parity gate failed: {receipt}")
    start = perf_counter()
    import_module("pcntoolkit")
    receipt["import_seconds"] = perf_counter() - start
    logger.info("imported pcntoolkit in %.2f seconds; starting 512 columns spread across the mask", receipt["import_seconds"])
    start = perf_counter()
    fit_groups(gm, cov, ref, np.linspace(0, SHAPE[1] - 1, CALIB_VOXELS, dtype=int), out / "pcntoolkit")
    receipt["seconds_512"] = perf_counter() - start
    receipt["sha256_reference"] = sha256(ARTIFACTS / CALIB_REF / "arrays.npz")
    receipt["passed"] = True
    with (out / "verified.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    logger.info("saved verified calibration %s: %s", out, json.dumps(receipt))


def shard_bounds(index: int, shards: int) -> tuple[int, int]:
    """Partition every C-order mask column exactly once."""
    if not (0 <= index < shards <= SHAPE[1]):
        raise ValueError(f"invalid shard {index}/{shards}")
    return index * SHAPE[1] // shards, (index + 1) * SHAPE[1] // shards


def check_job(value: str) -> str:
    """Accept only a concrete scheduler job identifier."""
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"invalid Slurm id {value!r}")
    return value


def read_calibration(results: Path, job: str) -> dict:
    """Require passed calibration from the explicitly bound job and reference."""
    path = results / "calibration" / check_job(job) / "verified.json"
    receipt = json.loads(path.read_text())
    source = resolve(CALIB_REF, "arrays.npz")["arrays.npz"]
    if (receipt["passed"], receipt["job"], receipt["sha256_reference"], receipt["epsilon"]) != (True, job, sha256(source), EPSILON):
        raise ValueError(f"invalid calibration receipt: {path}")
    return receipt


def fit_shard(results: Path, calibration: str, array: str, index: int, shards: int) -> None:
    """Fit and atomically publish exactly the assigned mask column range."""
    lo, hi = shard_bounds(index, shards)
    out = results / "arrays" / check_job(array)
    payload = out / f"shard-{index:03d}.npz"
    if payload.exists():
        raise FileExistsError(payload)
    verified = read_calibration(results, calibration)
    ref = reference(PCNTOOLKIT_REFERENCE)
    gm, cov, matrix = inputs(ref, verified)
    logger.info("array %s shard %d/%d, mask columns [%d,%d) from %s", array, index, shards, lo, hi, matrix["path"])
    voxels = np.arange(lo, hi)
    z, weights, nlz = fit_groups(gm, cov, ref, voxels, out / "pcntoolkit" / f"{index:03d}")
    write_new(payload, z_test=z, posterior_weights=weights, nlZ=nlz, voxels=voxels, test=ref["test"],
              array=array, calibration=calibration, shards=shards)
    logger.info("saved %s: %d voxels", payload, len(voxels))
