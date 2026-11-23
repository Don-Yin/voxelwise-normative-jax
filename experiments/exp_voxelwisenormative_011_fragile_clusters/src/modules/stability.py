"""Stability of every cluster of one person's single-fit map against that person's 64 refit flag maps."""
from __future__ import annotations

import logging

import numpy as np

from .clusters import ClusterGrid
from .inputs import THRESHOLD

logger = logging.getLogger(__name__)
CONNECTIVITIES = (26, 18, 6)  # neighbours: the main analysis (FSL's default), then two sensitivity analyses (SPM's default, AFNI's)
MAX_VOXELS = np.iinfo(np.uint16).max  # per-refit voxel counts are stored as uint16


def cropped(mask: np.ndarray) -> np.ndarray:
    """The mask inside its bounding box: the same true cells in the same C order, in a smaller volume to label."""
    cells = np.argwhere(mask)
    low, high = cells.min(axis=0), cells.max(axis=0) + 1
    return mask[tuple(slice(a, b) for a, b in zip(low, high, strict=True))]


def grids(mask: np.ndarray) -> dict[int, ClusterGrid]:
    """One labelling grid per connectivity, over the cropped mask."""
    small = cropped(mask)
    return {neighbours: ClusterGrid(small, (), neighbours) for neighbours in CONNECTIVITIES}


def best_overlap(clusters: np.ndarray, n_clusters: int, refit: np.ndarray, n_refit: int) -> np.ndarray:
    """Per cluster (labels 1 to n_clusters), its largest Jaccard overlap with any component of one refit's flag map."""
    best = np.zeros(n_clusters + 1)
    both = (clusters > 0) & (refit > 0)
    if both.any():
        sizes, other = np.bincount(clusters, minlength=n_clusters + 1), np.bincount(refit, minlength=n_refit + 1)
        pairs, shared = np.unique(clusters[both].astype(np.int64) * (n_refit + 1) + refit[both], return_counts=True)
        mine, theirs = np.divmod(pairs, n_refit + 1)
        np.maximum.at(best, mine, shared / (sizes[mine] + other[theirs] - shared))
    return best[1:]


def person_columns(grid: ClusterGrid, z: np.ndarray, refits: np.ndarray, threshold: float = THRESHOLD) -> dict[str, np.ndarray]:
    """One row per cluster of the single-fit map (positive clusters first, then negative), measured against every refit.

    `z` is the single fit's signed z (voxels,), `refits` every refit's flags at the same cutoff (refits, voxels). A refit's
    flags are unsigned, so its components may join a positive and a negative voxel; the single fit's clusters never do.
    `recurrence_sum` adds up, over a cluster's voxels, the number of refits that flag the voxel.
    """
    n_refits = len(refits)
    if z.shape != (grid.n,) or refits.shape != (n_refits, grid.n) or refits.dtype != bool:
        raise ValueError(f"inconsistent person: {z.shape=}, {refits.shape=}, {refits.dtype=}, {grid.n=}")
    counts = refits.sum(axis=0, dtype=np.uint8)
    positive, n_positive = grid.components(z > threshold)
    negative, n_negative = grid.components(z < -threshold)
    clusters = np.where(negative > 0, negative + n_positive, positive)
    n = n_positive + n_negative
    sizes = np.bincount(clusters, minlength=n + 1)[1:]
    if n and sizes.max() > MAX_VOXELS:
        raise ValueError(f"a cluster of {sizes.max()} voxels does not fit the uint16 per-refit counts")
    inside, jaccard = np.zeros((n, n_refits), dtype=np.uint16), np.zeros((n, n_refits))
    for r in range(n_refits if n else 0):
        inside[:, r] = np.bincount(clusters[refits[r]], minlength=n + 1)[1:]
        other, n_other = grid.components(refits[r])
        jaccard[:, r] = best_overlap(clusters, n, other, n_other)
    return {"sign": np.repeat(np.array([1, -1], dtype=np.int8), [n_positive, n_negative]),
            "voxels": sizes.astype(np.int32),
            "core_voxels": np.bincount(clusters[counts == n_refits], minlength=n + 1)[1:].astype(np.int32),
            "recurrence_sum": np.bincount(clusters, weights=counts, minlength=n + 1)[1:].astype(np.int32),
            "refit_voxels": inside, "refit_jaccard": jaccard.astype(np.float32), "jaccard_mean": jaccard.mean(axis=1)}


def fold_columns(grid_by_connectivity: dict[int, ClusterGrid], z: np.ndarray, packed: np.ndarray, label: str,
                 threshold: float = THRESHOLD) -> dict[int, dict[str, np.ndarray]]:
    """Cluster rows of every person of one fold, per connectivity, with each cluster's `person` (row within the fold)."""
    n_people = packed.shape[1]
    parts: dict[int, list[dict[str, np.ndarray]]] = {neighbours: [] for neighbours in grid_by_connectivity}
    for p in range(n_people):
        refits = np.unpackbits(packed[:, p, :], axis=-1, count=z.shape[1], bitorder="little").view(bool)
        for neighbours, grid in grid_by_connectivity.items():
            columns = person_columns(grid, z[p], refits, threshold)
            parts[neighbours].append({**columns, "person": np.full(len(columns["voxels"]), p, dtype=np.int32)})
        if (p + 1) % 10 == 0 or p + 1 == n_people:
            logger.info("[%d/%d] %s", p + 1, n_people, label)
    return {neighbours: {key: np.concatenate([part[key] for part in rows]) for key in rows[0]}
            for neighbours, rows in parts.items()}
