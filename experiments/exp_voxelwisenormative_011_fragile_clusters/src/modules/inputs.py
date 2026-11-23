"""Pinned IXI split and exp_005 reference draws; no experiment's result tree is an input."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from shared.voxelwisenormative.artifacts import VOXEL_MATRIX, resolve, sha256
from shared.voxelwisenormative.basis import Basis
from shared.voxelwisenormative.cohort import IXI_SITES
from shared.voxelwisenormative.ixi_folds import (
    DRAW_DIGEST,
    N_FOLDS,
    N_REFITS,
    N_TEST,
    N_TRAIN,
    N_VOXELS,
    SEED,
    THRESHOLD,
    fold_split,
    read_ixi_fold,
)
from shared.voxelwisenormative.ixi_folds import (
    draws as draws_for_rows,
)

__all__ = [
    "AXIAL_GRID", "DRAW_DIGEST", "MIN_HITS", "N_FOLDS", "N_REFITS", "N_SCANS", "N_TEST", "N_TRAIN",
    "N_VOXELS", "SEED", "THRESHOLD", "Cohort", "draws", "fold_split", "read_ixi", "read_mask", "synthetic",
]

N_SCANS = N_TRAIN + N_TEST
MIN_HITS = 52
AXIAL_GRID = Path(__file__).with_name("reader-facing-axial-grid.json")


@dataclass(frozen=True)
class Cohort:
    """Fixed response and covariate matrices in reference and held-out order."""

    train: np.ndarray
    test: np.ndarray
    phi_train: np.ndarray
    phi_test: np.ndarray
    reference_sites: np.ndarray
    train_rows: np.ndarray
    test_rows: np.ndarray
    mask: np.ndarray
    axial_indices: tuple[int, ...]
    axial_mm: tuple[float, ...]
    reference_ages: np.ndarray  # years, in reference order; only --reference-n reads it


def draws(cohort: Cohort, *, verify_original: bool) -> tuple[np.ndarray, str]:
    """Generate the archived recipe for these cohort rows."""
    return draws_for_rows(cohort.train_rows, verify_original=verify_original)


def read_mask(path: Path) -> tuple[np.ndarray, tuple[int, ...], tuple[float, ...]]:
    """Verify the atlas grid against the canonical slice receipt generated for this mask."""
    image = nib.load(path)
    if not isinstance(image, nib.Nifti1Image) or image.affine is None:
        raise TypeError(f"expected a NIfTI-1 image with a world affine: {path}")
    try:
        receipt = json.loads(AXIAL_GRID.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read the canonical slice receipt: {AXIAL_GRID}") from exc
    values = np.asarray(image.dataobj)
    mask = values != 0
    indices = tuple(receipt["resolved_voxel_indices"])
    levels = tuple(receipt["resolved_mni_z_mm"])
    if (sha256(path) != receipt["image_sha256"] or list(image.shape) != receipt["grid_shape"]
            or not np.array_equal(image.affine, receipt["grid_affine"])):
        raise ValueError(f"{path} does not match the canonical axial grid receipt {AXIAL_GRID}")
    if not np.isfinite(values).all() or mask.sum() != N_VOXELS or not mask[:, :, indices].any(axis=(0, 1)).all():
        raise ValueError(f"unexpected mask support in {path}: {mask.sum()} voxels, {indices=}")
    return mask, indices, levels


def read_ixi(fold: int = 0) -> Cohort:
    """Open the frozen Slurm inputs only after the caller has checked the CUDA job; see fold_split for folds."""
    mask, indices, levels = read_mask(resolve(VOXEL_MATRIX, "mask.nii.gz")["mask.nii.gz"])
    data = read_ixi_fold(fold, mask=mask)
    return Cohort(data.train, data.test, data.phi_train, data.phi_test, data.reference_sites,
                  data.train_rows, data.test_rows, mask, indices, levels, data.reference_ages)


def synthetic(fold: int = 0) -> Cohort:
    """Deterministic shaped scans with scanner and response shifts; never opens IXI files."""
    rng = np.random.default_rng(41)
    mask = np.ones((6, 7, 8), dtype=bool)
    mask[0, :2, :] = False
    n, held, v = 24, 4, int(mask.sum())
    site = np.array(IXI_SITES * 8 + ("Guys", "HH", "IOP", "Guys"))
    age = rng.uniform(24, 73, n + held)
    male = np.tile([False, True], (n + held) // 2)
    x = np.column_stack((age, male))
    basis = Basis.fit(x[:n], site[:n], batch_levels=IXI_SITES)
    phi = basis.apply(x, site)
    values = rng.normal(size=(n + held, v)).astype(np.float32)
    values[site == "Guys", :27] += 1.1
    region = np.zeros_like(mask)
    region[2:5, 3:5, 2:6] = True
    values[n, region[mask]] += 1.85
    values[n + 1, region[mask]] -= 2.3
    train, test = fold_split(np.arange(n + held) >= n, fold)
    return Cohort(values[train], values[test], phi[train], phi[test], site[train], train, test,
                  mask, (1, 3, 5), (1.0, 3.0, 5.0), age[train])
