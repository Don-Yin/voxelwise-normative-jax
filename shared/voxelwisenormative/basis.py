"""Training-scaled spline and fixed batch effects for a numeric covariate matrix."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


def _matrix(covariates) -> np.ndarray:
    """Convert non-empty, finite covariate rows to float64."""
    x = np.asarray(covariates, dtype=np.float64)
    if x.ndim != 2 or not all(x.shape) or not np.isfinite(x).all():
        raise ValueError(f"covariates must be a non-empty, finite 2D matrix; got shape {x.shape}")
    return x


def _batch_labels(batch, n_rows: int) -> tuple:
    """Read one scalar batch label per row, or no labels for an ungrouped design."""
    if batch is None:
        return ()
    labels = np.asarray(batch, dtype=object)
    if labels.shape != (n_rows,):
        raise ValueError(f"batch labels must have shape {(n_rows,)}, got {labels.shape}")
    return tuple(labels)


@dataclass(frozen=True)
class Basis:
    """A cubic spline, linear covariates, intercept and optional one-hot batch columns."""

    mean: np.ndarray
    sd: np.ndarray
    knots: np.ndarray
    spline_column: int
    batch_levels: tuple

    @classmethod
    def fit(cls, covariates, batch=None, *, spline_column: int = 0, batch_levels=None) -> Basis:
        """Learn scaling and padded knots on training rows; learn batch levels in first-seen order if omitted."""
        x = _matrix(covariates)
        if not isinstance(spline_column, (int, np.integer)) or not 0 <= spline_column < x.shape[1]:
            raise ValueError(f"spline_column {spline_column!r} is outside the {x.shape[1]} covariate columns")
        labels = _batch_labels(batch, len(x))
        levels = tuple(dict.fromkeys(labels)) if batch_levels is None else tuple(batch_levels)
        if len(set(levels)) != len(levels) or any(level is None or level != level for level in levels):
            raise ValueError(f"batch_levels must be unique and non-missing; got {levels!r}")
        if levels and not labels:
            raise ValueError("batch labels are required when batch levels are given")
        unknown = set(labels).difference(levels)
        if unknown:
            raise ValueError(f"unknown batch labels in training rows: {unknown!r}")

        mean, sd = x.mean(axis=0), x.std(axis=0)
        bad = np.flatnonzero(~np.isfinite(mean) | ~np.isfinite(sd) | (sd == 0))
        if bad.size:
            raise ValueError(f"invalid training scale at covariate column {bad[0]}")
        scaled = (x[:, spline_column] - mean[spline_column]) / sd[spline_column]
        padding = 0.05 * np.ptp(scaled)
        if padding == 0:
            raise ValueError(f"constant spline covariate at column {spline_column}")
        lo, hi = scaled.min() - padding, scaled.max() + padding
        knots = np.r_[np.repeat(lo, 3), np.linspace(lo, hi, 5), np.repeat(hi, 3)]
        return cls(mean, sd, knots, spline_column, levels)

    def apply(self, covariates, batch=None) -> np.ndarray:
        """Apply the training basis, extrapolating the spline and rejecting unknown batch labels."""
        x = _matrix(covariates)
        if x.shape[1] != self.mean.size:
            raise ValueError(f"expected {self.mean.size} covariate columns, got {x.shape[1]}")
        x = (x - self.mean) / self.sd
        if not np.isfinite(x).all():
            raise ValueError("non-finite covariates after applying training scaling")
        labels = _batch_labels(batch, len(x))
        if self.batch_levels and not labels:
            raise ValueError("batch labels are required by the fitted basis")
        categories = np.zeros((len(x), len(self.batch_levels)), dtype=bool)
        if labels:
            indices = {level: i for i, level in enumerate(self.batch_levels)}
            try:
                columns = [indices[label] for label in labels]
            except KeyError as exc:
                raise ValueError(f"unknown batch label {exc.args[0]!r}") from None
            categories[np.arange(len(x)), columns] = True

        i = self.spline_column
        spline = BSpline.design_matrix(x[:, i], self.knots, 3, extrapolate=True).toarray()
        return np.column_stack((spline, x[:, :i], x[:, i + 1:], np.ones(len(x)), categories))
