"""Verified IXI cohort inputs and train-only response scaling."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .artifacts import VOXEL_MATRIX, resolve
from .basis import Basis

IXI_SITES = ("HH", "Guys", "IOP")  # trailing column order of the frozen PCNtoolkit design


def load_covariates(*, root: Path | None = None) -> dict[str, np.ndarray]:
    """subject, age, sex and site columns of the voxel matrix rows."""
    path = resolve(VOXEL_MATRIX, "covariates.tsv", root=root)["covariates.tsv"]
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        "subject": np.array([r["subject"] for r in rows]),
        "age": np.array([float(r["age"]) for r in rows]),
        "sex": np.array([r["sex"] for r in rows]),
        "site": np.array([r["site"] for r in rows]),
    }


def load_grey_matter(mmap: bool = False, *, root: Path | None = None) -> np.ndarray:
    """float32 (subjects, voxels) grey matter matrix, verified against the manifest."""
    path = resolve(VOXEL_MATRIX, "gm.npy", root=root)["gm.npy"]
    return np.load(path, mmap_mode="r" if mmap else None)


def site_split(sites: np.ndarray, seed: int, test_fraction: float = 0.2) -> np.ndarray:
    """boolean test mask, stratified by site; seed 0 is the benchmark split."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(sites), bool)
    for site in sorted(set(sites)):
        idx = np.flatnonzero(sites == site)
        test[rng.choice(idx, round(test_fraction * len(idx)), replace=False)] = True
    return test


def ixi_covariates(cov: dict[str, np.ndarray]) -> np.ndarray:
    """Encode the IXI age and male-indicator columns used in the frozen design."""
    return np.column_stack((cov["age"], cov["sex"] == "M"))


def pinned_agreement(cov, reference) -> float:
    """Check the original IXI split's design against the archived PCNtoolkit arrays."""
    test = reference["test"]
    x, sites = ixi_covariates(cov), cov["site"]
    if test.dtype != bool or test.shape != (len(x),) or not test.any() or test.all():
        raise ValueError(f"invalid frozen IXI split: {test.shape=}, {test.dtype=}")
    observed = set(sites[~test])
    if observed != set(IXI_SITES):
        raise ValueError(f"expected training IXI sites {IXI_SITES}, got {observed}")
    basis = Basis.fit(x[~test], sites[~test], batch_levels=IXI_SITES)
    phi = basis.apply(x, sites)
    errors = []
    for key, rows in (("Phi_train", ~test), ("Phi_test", test)):
        actual, expected = phi[rows], reference[key]
        if actual.shape != expected.shape:
            raise ValueError(f"{key} has shape {expected.shape}, reproduced shape is {actual.shape}")
        errors.append(np.max(np.abs(expected - actual)))
    error = float(np.max(errors))
    if not error <= 1e-12:
        raise ValueError(f"reproduced covariate basis differs from experiment 002 by {error:g}")
    return error


def standardise(responses: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale float64 train and test responses by the training mean and population standard deviation."""
    train = responses[~test].astype(np.float64)
    mean, sd = train.mean(0), train.std(0)
    return (train - mean) / sd, (responses[test].astype(np.float64) - mean) / sd
