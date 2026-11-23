"""A deliberately plain re-implementation of the per-cluster statistics (flood fill over coordinate sets), to check the fast one against."""
from __future__ import annotations

import numpy as np

THRESHOLD = 1.96


def offsets(connectivity: int) -> list[tuple[int, int, int]]:
    """Neighbour steps: 6 share a face, 18 a face or an edge, 26 a face, an edge or a corner."""
    reach = {6: 1, 18: 2, 26: 3}[connectivity]
    return [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
            if 0 < abs(x) + abs(y) + abs(z) <= reach]


def components(cells: set[tuple[int, int, int]], steps: list[tuple[int, int, int]]) -> list[set[tuple[int, int, int]]]:
    """Connected components of a set of voxel coordinates."""
    seen: set[tuple[int, int, int]] = set()
    found = []
    for start in sorted(cells):
        if start in seen:
            continue
        seen.add(start)
        stack, part = [start], {start}
        while stack:
            x, y, z = stack.pop()
            for dx, dy, dz in steps:
                near = (x + dx, y + dy, z + dz)
                if near in cells and near not in seen:
                    seen.add(near)
                    part.add(near)
                    stack.append(near)
        found.append(part)
    return found


def person(mask: np.ndarray, z: np.ndarray, refits: np.ndarray, counts: np.ndarray, connectivity: int,
           threshold: float = THRESHOLD) -> dict[tuple[int, int], dict]:
    """Statistics of each cluster, keyed by (sign, first mask column of the cluster): the same facts as stability.person_columns."""
    coords = [tuple(int(i) for i in c) for c in np.argwhere(mask)]
    column = {c: v for v, c in enumerate(coords)}
    steps = offsets(connectivity)

    def as_cells(flagged: np.ndarray) -> set[tuple[int, int, int]]:
        return {coords[v] for v in np.flatnonzero(flagged)}

    refit_parts = [components(as_cells(flags), steps) for flags in refits]
    out = {}
    for sign, flags in ((1, z > threshold), (-1, z < -threshold)):
        for part in components(as_cells(flags), steps):
            first = min(column[c] for c in part)
            jaccard = []
            for parts in refit_parts:
                jaccard.append(max((len(part & other) / len(part | other) for other in parts), default=0.0))
            out[(sign, first)] = {
                "voxels": len(part),
                "core_voxels": sum(1 for c in part if counts[column[c]] == len(refits)),
                "refit_voxels": [sum(1 for c in part if flags_r[column[c]]) for flags_r in refits],
                "refit_jaccard": jaccard, "jaccard_mean": float(np.mean(jaccard)),
            }
    return out
