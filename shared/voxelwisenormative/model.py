"""Fit and score arbitrary response columns with the shared warped regression."""
from __future__ import annotations

import numpy as np

from .basis import Basis
from .batched_blr import design, fit_chunks, make_fitter
from .cohort import standardise


def fit_score(covariates, responses, held_out, batch=None, *, spline_column: int = 0,
              batch_levels=None, chunk: int = 32768) -> np.ndarray:
    """Fit on rows outside the boolean mask, returning held-out z-scores in response-column order."""
    x, y, held = map(np.asarray, (covariates, responses, held_out))
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not y.shape[1]:
        raise ValueError(f"need covariate and response matrices with the same rows and at least one response: {x.shape=}, {y.shape=}")
    if held.dtype != bool or held.shape != (len(x),) or not held.any() or np.count_nonzero(~held) < 2:
        raise ValueError(f"held_out must be a row-aligned boolean mask with at least two training rows and one test row; "
                         f"got {held.shape=}, {held.dtype=}")
    if not isinstance(chunk, (int, np.integer)) or chunk < 1:
        raise ValueError(f"chunk must be a positive integer, got {chunk!r}")
    if batch is not None:
        batch = np.asarray(batch, dtype=object)
        if batch.shape != held.shape:
            raise ValueError(f"batch labels must have shape {held.shape}, got {batch.shape}")

    basis = Basis.fit(x[~held], None if batch is None else batch[~held],
                      spline_column=spline_column, batch_levels=batch_levels)
    phi = basis.apply(x, batch)
    with np.errstate(divide="raise", invalid="raise", over="raise"):
        try:
            train, test = standardise(y, held)
        except FloatingPointError as exc:
            raise ValueError(f"responses must be finite with positive training variance: {exc}") from exc
    if not (np.isfinite(train).all() and np.isfinite(test).all()):
        raise ValueError("non-finite responses after applying training scaling")
    return fit_chunks(make_fitter(), design(phi[~held], phi[held]), train, test, chunk)[2]
