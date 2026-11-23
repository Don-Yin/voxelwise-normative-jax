"""the IXI cohort: verified grey matter matrix, covariates, site-stratified split and scaling."""

from __future__ import annotations

import csv

import numpy as np

from .artifacts import VOXEL_MATRIX, resolve


def load_covariates() -> dict[str, np.ndarray]:
    """subject, age, sex and site columns of the voxel matrix rows."""
    path = resolve(VOXEL_MATRIX, "covariates.tsv")["covariates.tsv"]
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        "subject": np.array([r["subject"] for r in rows]),
        "age": np.array([float(r["age"]) for r in rows]),
        "sex": np.array([r["sex"] for r in rows]),
        "site": np.array([r["site"] for r in rows]),
    }


def load_grey_matter(mmap: bool = False) -> np.ndarray:
    """float32 (subjects, voxels) grey matter matrix, verified against the manifest."""
    path = resolve(VOXEL_MATRIX, "gm.npy")["gm.npy"]
    return np.load(path, mmap_mode="r" if mmap else None)


def site_split(sites: np.ndarray, seed: int, test_fraction: float = 0.2) -> np.ndarray:
    """boolean test mask, stratified by site; seed 0 is the benchmark split."""
    rng = np.random.default_rng(seed)
    test = np.zeros(len(sites), bool)
    for site in sorted(set(sites)):
        idx = np.flatnonzero(sites == site)
        test[rng.choice(idx, round(test_fraction * len(idx)), replace=False)] = True
    return test


def standardise(gm: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """train and test float64 values scaled by the training mean and sd (ddof 0, as PCNtoolkit)."""
    train = gm[~test].astype(np.float64)
    mean, sd = train.mean(0), train.std(0)
    return (train - mean) / sd, (gm[test].astype(np.float64) - mean) / sd
