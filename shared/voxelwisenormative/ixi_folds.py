"""Pinned public IXI folds, covariate design and bootstrap row selection."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np

from .artifacts import PCNTOOLKIT_REFERENCE, VOXEL_MATRIX, resolve
from .cohort import load_covariates, pinned_agreement, standardise
from .template_grid import TemplateGrid

logger = logging.getLogger(__name__)
N_TRAIN, N_TEST, N_VOXELS = 450, 113, 165_402
N_REFITS, N_FOLDS = 64, 5
THRESHOLD, SEED = 1.96, 20261121
# SHA-256 of experiment 005's sampled cohort rows (C-order, little-endian int64).
DRAW_DIGEST = "6ade730b75ff6977f116062b24d9258621148d237dd6df301787da0183807f31"


@dataclass(frozen=True)
class IXIFold:
    """Public response and design matrices in cohort row order."""

    train: np.ndarray
    test: np.ndarray
    phi_train: np.ndarray
    phi_test: np.ndarray
    reference_sites: np.ndarray
    train_rows: np.ndarray
    test_rows: np.ndarray
    mask: np.ndarray
    reference_ages: np.ndarray


def fold_split(held: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Fold 0 holds out the frozen split; the others split its original reference rows."""
    if held.dtype != bool or not 0 <= fold < N_FOLDS:
        raise ValueError(f"expected a boolean held-out mask and a fold from 0 to {N_FOLDS - 1}, got {held.dtype=}, {fold=}")
    test = np.flatnonzero(held)
    if fold > 0:
        parts = np.array_split(np.random.default_rng(SEED).permutation(np.flatnonzero(~held)), N_FOLDS - 1)
        test = np.sort(parts[fold - 1])
    return np.setdiff1d(np.arange(held.size), test), test


def row_digest(rows: np.ndarray) -> str:
    """Hash cohort rows in C-order, little-endian int64."""
    return hashlib.sha256(np.ascontiguousarray(rows, dtype="<i8").tobytes()).hexdigest()


def draws(train_rows: np.ndarray, *, verify_original: bool) -> tuple[np.ndarray, str]:
    """Generate ordered reference indices with replacement; check the archived digest when requested."""
    n = len(train_rows)
    indices = np.random.default_rng(SEED).integers(0, n, size=(N_REFITS, n), dtype=np.int32)
    digest = row_digest(train_rows[indices])
    if verify_original and digest != DRAW_DIGEST:
        raise ValueError(f"exp_005 reference draw digest {digest} differs from {DRAW_DIGEST}")
    return indices, digest


def read_ixi_fold(fold: int = 0, *, mask: np.ndarray | None = None) -> IXIFold:
    """Read the manifest-checked public IXI fold and the frozen PCNtoolkit design."""
    gm_path = resolve(VOXEL_MATRIX, "gm.npy")["gm.npy"]
    if mask is None:
        mask = TemplateGrid().mask
    gm = np.load(gm_path, mmap_mode="r", allow_pickle=False)
    cov = load_covariates()
    expected = (N_TRAIN + N_TEST, N_VOXELS)
    if (gm.shape != expected or gm.dtype != np.float32 or len(set(cov["subject"])) != expected[0]
            or mask.sum() != N_VOXELS):
        raise ValueError(f"unexpected IXI matrix/cohort: {gm_path}, {gm.shape=}, {gm.dtype=}, {mask.sum()=}")
    path = resolve(PCNTOOLKIT_REFERENCE, "arrays.npz")["arrays.npz"]
    with np.load(path, allow_pickle=False) as reference:
        held = reference["test"]
        if held.dtype != bool or held.shape != (expected[0],) or held.sum() != N_TEST:
            raise ValueError(f"unexpected frozen held-out split in {path}: {held.shape=}, {held.dtype=}")
        err = pinned_agreement(cov, reference)
        y, yt = standardise(gm[:, reference["voxels"]], held)
        diff = max(np.max(np.abs(y - reference["y_train"])), np.max(np.abs(yt - reference["y_test"])))
        if not np.isfinite(diff) or diff > 1e-9:
            raise ValueError(f"{gm_path} row/column/scaling error against {path}: {diff}")
        phi = np.empty((held.size, reference["Phi_train"].shape[1]), dtype=reference["Phi_train"].dtype)
        phi[~held], phi[held] = reference["Phi_train"], reference["Phi_test"]
    logger.info("verified frozen split %d reference, %d held out, basis difference %.3g", N_TRAIN, N_TEST, err)
    train, test = fold_split(held, fold)
    logger.info("fold %d: %d reference, %d held out", fold, len(train), len(test))
    return IXIFold(gm[train], gm[test], phi[train], phi[test], cov["site"][train], train, test, mask, cov["age"][train])
