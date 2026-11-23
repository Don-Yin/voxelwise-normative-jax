"""Standardise each training selection and fit it only in a guarded A100 job."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .batch_job import require_batch_job
from .batched_blr import Design, design, fit_chunks, make_fitter

Score = Callable[[Design, np.ndarray, np.ndarray], np.ndarray]


def scores_for(phi_train: np.ndarray, raw_train: np.ndarray, phi_test: np.ndarray,
               raw_test: np.ndarray, scorer: Score) -> np.ndarray:
    """Scale the query by the selected training mean and population SD, then score."""
    train = raw_train.astype(np.float64)
    mean, sd = train.mean(axis=0), train.std(axis=0)
    bad = ~np.isfinite(sd) | (sd <= 0) | ~np.isfinite(mean)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} invalid response scales, first column {int(np.flatnonzero(bad)[0])}")
    train -= mean
    train /= sd
    test = (raw_test - mean) / sd
    z = scorer(design(phi_train, phi_test), train, test)
    if z.shape != raw_test.shape or not np.isfinite(z).all():
        raise ValueError(f"invalid held-out scores: {z.shape=}, expected {raw_test.shape}")
    return z


def cuda_scorer(chunk: int, *, save_hyp: Callable[[np.ndarray], None] | None = None,
                fit_factory=make_fitter) -> tuple[Score, dict[str, str | None]]:
    """Require an A100 batch job; optionally capture validated fitted hyperparameters."""
    batch = require_batch_job("real IXI input")
    import jax

    device = jax.devices()[0]
    if device.platform != "gpu" or "A100" not in str(device.device_kind):
        raise RuntimeError(f"an A100 CUDA allocation is required, JAX selected {device}")
    fit = fit_factory()

    def score(rotated: Design, y: np.ndarray, yt: np.ndarray) -> np.ndarray:
        """Fit each voxel and reject nonfinite optimiser outputs."""
        hyp, objective, z = fit_chunks(fit, rotated, y, yt, chunk)
        if not np.isfinite(hyp).all() or not np.isfinite(objective).all():
            raise ValueError("batched warped regression returned non-finite hyperparameters or evidence")
        if save_hyp is not None:
            save_hyp(hyp)
        return z

    ids = {"slurm_job": batch.job_id} if batch.platform == "slurm" else {"slurm_job": None, "hugging_face_job": batch.job_id}
    return score, {"device": str(device.device_kind), "jax": jax.__version__, **ids}
