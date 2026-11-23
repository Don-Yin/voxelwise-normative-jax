"""Case bootstrap over the original IXI reference subjects; never refit on the held-out row."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from shared.voxelwisenormative.artifacts import (
    PCNTOOLKIT_REFERENCE,
    VOXEL_MATRIX,
    resolve,
)
from shared.voxelwisenormative.batched_blr import design, fit_chunks, make_fitter
from shared.voxelwisenormative.cohort import standardise

from .slice_view import N_VOXELS, slice_columns

logger = logging.getLogger(__name__)
TEST_ROW = 0  # subject 1 on the cover slide; chosen before the bootstrap
N_TRAIN, N_TEST = 450, 113
Z_THRESHOLD = 1.96
SURVIVAL_NUMERATOR, SURVIVAL_DENOMINATOR = 4, 5


@dataclass(frozen=True)
class Inputs:
    """Raw grey matter, frozen basis rows and the one display slice."""

    train: np.ndarray
    held_out: np.ndarray
    phi_train: np.ndarray
    phi_test: np.ndarray
    train_cohort_rows: np.ndarray
    held_out_cohort_row: int
    columns: np.ndarray


def load_inputs() -> Inputs:
    """Verify the pinned input split and column order against its converged reference subset."""
    files = resolve(VOXEL_MATRIX, "gm.npy", "mask.nii.gz")
    columns = slice_columns(files["mask.nii.gz"])
    raw = np.load(files["gm.npy"], mmap_mode="r")
    with np.load(resolve(PCNTOOLKIT_REFERENCE, "arrays.npz")["arrays.npz"]) as ref:
        test = ref["test"]
        rows, held = np.flatnonzero(~test), np.flatnonzero(test)
        if (raw.shape != (N_TRAIN + N_TEST, N_VOXELS) or test.dtype != bool
                or test.shape != (N_TRAIN + N_TEST,) or (len(rows), len(held)) != (N_TRAIN, N_TEST)):
            raise ValueError(f"unexpected matrix or split: {raw.shape}, {test.shape}/{test.dtype}, {len(rows)}/{len(held)}")
        p, pt = ref["Phi_train"], ref["Phi_test"]
        if p.shape != (N_TRAIN, 12) or pt.shape != (N_TEST, 12):
            raise ValueError(f"unexpected pinned basis shapes: {p.shape}, {pt.shape}")
        y, yt = standardise(raw[:, ref["voxels"]], test)
        difference = max(np.max(np.abs(y - ref["y_train"])), np.max(np.abs(yt - ref["y_test"])))
        if not np.isfinite(difference) or difference > 1e-9:
            raise ValueError(f"fixed split/y scaling disagrees with converged reference by {difference}")
    return Inputs(raw[rows], raw[held[TEST_ROW]], p, pt[TEST_ROW:TEST_ROW+1], rows, int(held[TEST_ROW]), columns)


def bootstrap_rows(seed: int, trials: int, n_train: int) -> np.ndarray:
    """One size-n draw with replacement per trial; prefixes agree across trial counts."""
    return np.random.default_rng(seed).integers(0, n_train, size=(trials, n_train), dtype=np.int32)


def scale(y: np.ndarray, yt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use only this trial's training statistics (population SD), including duplicate rows."""
    y = y.astype(np.float64)
    mean, sd = y.mean(axis=0), y.std(axis=0)
    bad = ~np.isfinite(sd) | (sd <= 0) | ~np.isfinite(mean)
    if bad.any():
        raise ValueError(f"invalid training scale at {bad.sum()} voxels; first column {np.flatnonzero(bad)[0]}")
    y -= mean
    y /= sd
    return y, ((yt - mean) / sd)[None, :]


def score(data: Inputs, rows: np.ndarray, fitter, chunk: int) -> tuple[np.ndarray, float]:
    """Fit all voxel columns, predicting the same held-out row for every resample."""
    y, yt = scale(data.train[rows], data.held_out)
    d = design(data.phi_train[rows], data.phi_test)
    start = time.perf_counter()
    hyp, objective, scores = fit_chunks(fitter, d, y, yt, chunk)
    elapsed = time.perf_counter() - start
    if scores.shape != yt.shape or not all(np.isfinite(x).all() for x in (hyp, objective, scores)):
        raise ValueError(f"invalid full-mask fit: z shape {scores.shape} vs {yt.shape}, "
                         f"non-finite z={np.count_nonzero(~np.isfinite(scores))}")
    return scores[0], elapsed


def survives(baseline_flags: np.ndarray, counts: np.ndarray, trials: int) -> np.ndarray:
    """Only original flags eligible; equality at 80% survives without float rounding."""
    return baseline_flags & (counts * SURVIVAL_DENOMINATOR >= trials * SURVIVAL_NUMERATOR)


def gpu_fitter():
    """Refuse CPU fallback for measured and live runs."""
    import jax

    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise RuntimeError(f"a CUDA GPU is required for this whole-brain run; JAX sees {devices}")
    return make_fitter(), jax.__version__, str(devices[0].device_kind)
