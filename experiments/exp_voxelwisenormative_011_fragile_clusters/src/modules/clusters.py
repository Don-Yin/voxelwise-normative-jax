"""Three-dimensional face-connected clusters on the frozen C-order analysis mask."""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import ndimage

from .inputs import MIN_HITS, N_REFITS

type Row = dict[str, int | str]
# scipy's structure rank for each neighbour count: faces; faces and edges; faces, edges and corners
NEIGHBOURS = {6: 1, 18: 2, 26: 3}


@dataclass(frozen=True)
class Cluster:
    """Size and visibility of the largest component in seven locked axial planes."""

    voxels: int
    on_slices: tuple[int, ...]

    def slice_field(self) -> str:
        """Return the ordered integer list as a single tabular field."""
        return ",".join(map(str, self.on_slices))


class ClusterGrid:
    """Reuse one full-grid mask and component-label buffer across the held-out scans."""

    def __init__(self, mask: np.ndarray, axial_indices: tuple[int, ...], connectivity: int = 6):
        if mask.ndim != 3 or mask.dtype != bool or not mask.any():
            raise ValueError(f"cluster mask must be nonempty 3D bool, got {mask.shape=} and {mask.dtype=}")
        if any(k < 0 or k >= mask.shape[2] for k in axial_indices):
            raise ValueError(f"axial indices {axial_indices} outside mask depth {mask.shape[2]}")
        if connectivity not in NEIGHBOURS:
            raise ValueError(f"connectivity must be one of {sorted(NEIGHBOURS)} neighbours, got {connectivity}")
        self.mask, self.axial_indices = mask, axial_indices
        self.n = int(mask.sum())
        self.volume = np.zeros(mask.shape, dtype=bool)
        self.labels = np.zeros(mask.shape, dtype=np.int32)
        self.structure = ndimage.generate_binary_structure(3, NEIGHBOURS[connectivity])

    def components(self, row: np.ndarray) -> tuple[np.ndarray, int]:
        """Component label (0 = none) of every mask column of one boolean row, and the number of components."""
        if row.shape != (self.n,) or row.dtype != bool:
            raise ValueError(f"expected {self.n} boolean mask columns, got {row.shape=} and {row.dtype=}")
        self.volume[self.mask] = row
        n = ndimage.label(self.volume, structure=self.structure, output=self.labels)
        return self.labels[self.mask], cast(int, n)  # with an output array, scipy returns only the count

    def largest(self, row: np.ndarray) -> Cluster:
        """Pick the first C-order component when largest sizes tie."""
        labels, n = self.components(row)
        if not n:
            return Cluster(0, (0,) * len(self.axial_indices))
        sizes = np.bincount(labels)
        selected = 1 + int(np.argmax(sizes[1:]))
        return Cluster(int(sizes[selected]),
                       tuple(int(np.count_nonzero(self.labels[:, :, k] == selected)) for k in self.axial_indices))


def ranked(rows: list[Row], cluster_key: str, total_key: str, rank_key: str) -> list[Row]:
    """Rank by largest cluster, then flag total and held-out row; stable order breaks further ties."""
    rows = sorted(rows, key=lambda r: (-int(r[cluster_key]), -int(r[total_key]), int(r["held_out_row_zero_based"])))
    return [{rank_key: i, **row} for i, row in enumerate(rows, start=1)]


def bootstrap_ranking(counts: np.ndarray, first: np.ndarray, grid: ClusterGrid) -> list[Row]:
    """Rank clusters from refit 1 among voxels that occur in 1–51 of the 64 refits."""
    if (counts.dtype != np.uint8 or first.dtype != bool or counts.shape != first.shape
            or counts.ndim != 2 or counts.shape[1] != grid.n or counts.max() > N_REFITS
            or np.any(first & (counts == 0))):
        raise ValueError(f"inconsistent recurrence counts and refit-1 flags: {counts.shape=}, {first.shape=}")
    rows = []
    for i, (hits, observed) in enumerate(zip(counts, first, strict=True)):
        unstable = (hits > 0) & (hits < MIN_HITS)
        fragile = observed & unstable
        component = grid.largest(fragile)
        rows.append({"held_out_row_zero_based": i, "largest_cluster_voxels": component.voxels,
                     "first_unstable_voxels": int(fragile.sum()), "ever_unstable_voxels": int(unstable.sum()),
                     "largest_cluster_axial_voxels": component.slice_field()})
    return ranked(rows, "largest_cluster_voxels", "first_unstable_voxels", "rank")


def contrast_ranking(name: str, control: str, flags: np.ndarray, baseline: np.ndarray, grid: ClusterGrid) -> list[Row]:
    """Rank one scanner comparison, measuring gained and lost clusters separately."""
    if (flags.dtype != bool or baseline.dtype != bool or flags.shape != baseline.shape
            or flags.ndim != 2 or flags.shape[1] != grid.n):
        raise ValueError(f"inconsistent flags for {name} versus {control}: {flags.shape=}, {baseline.shape=}")
    rows = []
    for i, (curr, ref) in enumerate(zip(flags, baseline, strict=True)):
        gained, lost = curr & ~ref, ref & ~curr
        add, drop = grid.largest(gained), grid.largest(lost)
        rows.append({"contrast": name, "compared_with": control, "held_out_row_zero_based": i,
                     "largest_flip_cluster_voxels": max(add.voxels, drop.voxels),
                     "flipped_voxels": int(gained.sum() + lost.sum()),
                     "new_flags": int(gained.sum()), "removed_flags": int(lost.sum()),
                     "new_largest_cluster_voxels": add.voxels, "removed_largest_cluster_voxels": drop.voxels,
                     "new_largest_cluster_axial_voxels": add.slice_field(),
                     "removed_largest_cluster_axial_voxels": drop.slice_field()})
    return ranked(rows, "largest_flip_cluster_voxels", "flipped_voxels", "rank_within_contrast")
