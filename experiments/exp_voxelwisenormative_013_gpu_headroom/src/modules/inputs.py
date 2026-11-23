"""Frozen IXI 450/113 split with exp_005's 64 reference draws, or a tiny synthetic cohort; no result tree is read."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

import numpy as np

from shared.voxelwisenormative.artifacts import (
    PCNTOOLKIT_REFERENCE,
    VOXEL_MATRIX,
    resolve,
)
from shared.voxelwisenormative.basis import SITES, Basis, pinned_agreement
from shared.voxelwisenormative.batch_job import require_batch_job
from shared.voxelwisenormative.batched_blr import design
from shared.voxelwisenormative.cohort import load_covariates, standardise

logger = logging.getLogger(__name__)
N_TRAIN, N_TEST, N_VOXELS, N_REFITS = 450, 113, 165_402, 64
DRAW_SEED = 20261121  # exp_005's recipe, as exp_011 reuses it
# SHA-256 of the experiment 005 cohort rows (C-order, little-endian int64), as in experiment 011.
DRAW_DIGEST = "6ade730b75ff6977f116062b24d9258621148d237dd6df301787da0183807f31"


@dataclass(frozen=True)
class Inputs:
    """Raw float32 responses, every draw's rotated design, and the optional PCNtoolkit check."""

    train: np.ndarray  # (N, V) reference rows
    test: np.ndarray  # (Nt, V) held-out rows
    rows: np.ndarray  # (R, N) int32 draws into the reference rows
    lam: np.ndarray  # (R + 1, D); index R is the original fit on every reference row once
    G: np.ndarray  # (R + 1, D, N)
    H: np.ndarray  # (R + 1, Nt, D)
    draw_digest: str
    design_seconds: float
    pinned_voxels: np.ndarray | None = None
    pinned_z: np.ndarray | None = None


def draws(train_rows: np.ndarray, refits: int, *, verify: bool) -> tuple[np.ndarray, str]:
    """exp_005's draws with replacement; a full IXI run must reproduce its frozen digest."""
    n = len(train_rows)
    indices = np.random.default_rng(DRAW_SEED).integers(0, n, size=(N_REFITS, n), dtype=np.int32)
    digest = hashlib.sha256(train_rows[indices].astype("<i8", copy=False).tobytes()).hexdigest()
    if verify and digest != DRAW_DIGEST:
        raise ValueError(f"exp_005 reference draw digest {digest} differs from {DRAW_DIGEST}")
    return indices[:refits], digest


def rotated(phi_train: np.ndarray, phi_test: np.ndarray,
            rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Stack every draw's eigenbasis, then the original fit; returns lam, G, H and the seconds taken."""
    start = time.perf_counter()
    parts = [design(phi_train[r], phi_test) for r in (*rows, np.arange(len(phi_train)))]
    lam, G, H = (np.stack([getattr(d, key) for d in parts]) for key in ("lam", "G", "H"))
    return lam, G, H, time.perf_counter() - start


def read_ixi() -> Inputs:
    """Bind the verified IXI matrix and frozen split; refused outside a Linux Slurm job or a marked Hugging Face job."""
    require_batch_job("real IXI input")
    gm_path = resolve(VOXEL_MATRIX, "gm.npy")["gm.npy"]
    gm = np.load(gm_path, mmap_mode="r", allow_pickle=False)
    cov = load_covariates()
    if gm.shape != (N_TRAIN + N_TEST, N_VOXELS) or gm.dtype != np.float32 or len(set(cov["subject"])) != gm.shape[0]:
        raise ValueError(f"unexpected IXI matrix/cohort: {gm_path}, {gm.shape=}, {gm.dtype=}")
    path = resolve(PCNTOOLKIT_REFERENCE, "arrays.npz")["arrays.npz"]
    with np.load(path, allow_pickle=False) as ref:
        held = ref["test"]
        if held.dtype != bool or held.shape != (gm.shape[0],) or held.sum() != N_TEST:
            raise ValueError(f"unexpected frozen held-out split in {path}: {held.shape=}, {held.dtype=}")
        pinned_agreement(cov, ref)
        y, yt = standardise(gm[:, ref["voxels"]], held)
        diff = max(np.max(np.abs(y - ref["y_train"])), np.max(np.abs(yt - ref["y_test"])))
        if not np.isfinite(diff) or diff > 1e-9:
            raise ValueError(f"{gm_path} row/column/scaling error against {path}: {diff}")
        phi_train, phi_test, voxels, z = (ref[k] for k in ("Phi_train", "Phi_test", "voxels", "z_test"))
    train_rows, test_rows = np.flatnonzero(~held), np.flatnonzero(held)
    rows, digest = draws(train_rows, N_REFITS, verify=True)
    lam, G, H, seconds = rotated(phi_train, phi_test, rows)
    logger.info("reading %d reference and %d held-out rows of %s", len(train_rows), len(test_rows), gm_path)
    train, test = np.ascontiguousarray(gm[train_rows]), np.ascontiguousarray(gm[test_rows])
    logger.info("verified exp_005 draws %s; %d draw designs in %.3fs", digest[:12], len(rows), seconds)
    return Inputs(train, test, rows, lam, G, H, digest, seconds, voxels, z)


def synthetic(n_train: int = 40, n_test: int = 8, voxels: int = 300, refits: int = 8) -> Inputs:
    """Deterministic skewed responses with age, sex and site effects; never opens IXI files."""
    rng = np.random.default_rng(13)
    total = n_train + n_test
    age, male = rng.uniform(20, 86, total), rng.random(total) < 0.5
    sex, site = np.where(male, "M", "F"), np.resize(np.asarray(SITES), total)
    basis = Basis.fit(age[:n_train], sex[:n_train], site[:n_train])
    phi = basis.apply(age, sex, site)
    mean = 0.5 - 0.003 * (age - 50) + 0.02 * male + 0.015 * (site == SITES[0])
    values = (mean[:, None] + rng.uniform(0.03, 0.08, voxels) * rng.gamma(3.0, 1.0, (total, voxels))).astype(np.float32)
    rows, digest = draws(np.arange(n_train), refits, verify=False)
    lam, G, H, seconds = rotated(phi[:n_train], phi[n_train:], rows)
    return Inputs(values[:n_train], values[n_train:], rows, lam, G, H, digest, seconds)
