"""The frozen IXI split, design and accuracy gate shared by all timings."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from shared.voxelwisenormative.artifacts import PCNTOOLKIT_REFERENCE, resolve
from shared.voxelwisenormative.batched_blr import Design, design
from shared.voxelwisenormative.cohort import load_grey_matter, standardise

logger = logging.getLogger(__name__)
TOL = 0.01


@dataclass(frozen=True)
class Dataset:
    """normalized observations, rotated design and matching reference columns."""

    y: np.ndarray
    yt: np.ndarray
    design: Design
    idx: np.ndarray
    z_ref: np.ndarray
    objective_ref: np.ndarray


def load(mode: str) -> Dataset:
    """verify the input artifact and reproduce both halves of its data split."""
    path = resolve(PCNTOOLKIT_REFERENCE, "arrays.npz")["arrays.npz"]
    with np.load(path) as ref:
        d = design(ref["Phi_train"], ref["Phi_test"])
        y_ref, yt_ref, z_ref, objective_ref = (ref[key] for key in ("y_train", "y_test", "z_test", "nlZ"))
        if mode == "full":
            y, yt = standardise(load_grey_matter(), ref["test"])
            idx = ref["voxels"]
            if np.unique(idx).size != idx.size or np.any((idx < 0) | (idx >= y.shape[1])):
                raise ValueError(f"invalid reference voxel IDs in {path}: {idx.size=}, {idx.min()=}, {idx.max()=}")
            np.testing.assert_allclose(y[:, idx], y_ref, rtol=0, atol=1e-9, err_msg=f"train differs: {path}")
            np.testing.assert_allclose(yt[:, idx], yt_ref, rtol=0, atol=1e-9, err_msg=f"test differs: {path}")
        else:
            y, yt, idx = y_ref, yt_ref, np.arange(y_ref.shape[1])
    if z_ref.shape != yt_ref.shape or idx.size != 1000 or not np.isfinite(z_ref).all():
        raise ValueError(f"invalid reference z in {path}: {idx.size=}, {z_ref.shape=}, {yt_ref.shape=}")
    logger.info("%s: %d train, %d test, %d mask voxels, %d reference voxels", mode, *[y.shape[0], *yt.shape, idx.size])
    return Dataset(y, yt, d, idx, z_ref, objective_ref)


def agreement(data: Dataset, fv: np.ndarray, z: np.ndarray) -> dict:
    """gate against reference z-scores plus finite checks across the full mask."""
    if z.shape != data.yt.shape or fv.shape != (z.shape[1],):
        raise ValueError(f"prediction shape mismatch: {z.shape=}, {data.yt.shape=}, {fv.shape=}")
    dz = np.abs(z[:, data.idx] - data.z_ref)
    obj_diff = fv[data.idx] - data.objective_ref
    max_dz = float(dz.max()) if np.isfinite(dz).all() else None
    bad = z.size - int(np.isfinite(z).sum())
    return {
        "z_absdiff_max": max_dz,
        "nonfinite_z": bad,
        "objective_minus_pcntoolkit_max": float(obj_diff.max()) if np.isfinite(obj_diff).all() else None,
        "passed": bad == 0 and max_dz is not None and max_dz <= TOL,
    }
