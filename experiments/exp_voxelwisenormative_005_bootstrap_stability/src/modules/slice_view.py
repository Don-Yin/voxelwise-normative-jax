"""Cover-slide axial view: rank the frozen mask's C-order columns and keep its exact z palette."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

N_VOXELS = 165_402
AXIAL_INDEX = 48
TILE_WIDTH, TILE_HEIGHT = 122, 86
COLOURS = np.array([(39, 110, 180), (231, 232, 225), (186, 48, 55)])
BACKGROUND = (24, 36, 51)
DIMMED = (63, 77, 93)
Z_LIMIT = 3


def slice_columns(path: Path) -> np.ndarray:
    """Read the verified little-endian IXI grid; map one canonical slice to voxel columns."""
    blob = gzip.decompress(path.read_bytes())
    dim, *shape = struct.unpack_from("<4h", blob, 40)
    encoding = struct.unpack_from("<hh", blob, 70)
    affine = np.array(struct.unpack_from("<12f", blob, 280)).reshape(3, 4)
    expected = np.array([[-2, 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72]])
    if (blob[344:348] != b"n+1\0" or dim != 3 or shape != [91, 109, 91]
            or encoding != (16, 32) or not np.array_equal(affine, expected)):
        raise ValueError(f"unexpected IXI grid/layout in {path}: {shape}, {encoding}, {affine}")
    offset = int(struct.unpack_from("<f", blob, 108)[0])
    mask = np.frombuffer(blob, dtype="<f4", count=int(np.prod(shape)), offset=offset)
    mask = mask.reshape(shape, order="F") != 0
    if int(mask.sum()) != N_VOXELS:
        raise ValueError(f"{path} contains {mask.sum()} mask voxels; expected {N_VOXELS}")
    columns = np.full(shape, -1, dtype=np.int32)
    columns[mask] = np.arange(N_VOXELS)
    # The input grid is L/A/S. Reverse x for canonical R/A/S; north is anterior.
    plane = np.rot90(columns[::-1, :, AXIAL_INDEX])
    yy, xx = np.where(plane >= 0)
    core = plane[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
    h, w = core.shape
    if h > TILE_HEIGHT or w > TILE_WIDTH:
        raise ValueError(f"axial slice at {AXIAL_INDEX} does not fit {TILE_WIDTH}×{TILE_HEIGHT}: {core.shape}")
    tile = np.full((TILE_HEIGHT, TILE_WIDTH), -1, dtype=np.int32)
    x, y = (TILE_WIDTH - w) // 2, (TILE_HEIGHT - h) // 2
    tile[y:y+h, x:x+w] = core
    return tile


def colours(z: np.ndarray) -> np.ndarray:
    """Colour full-reference z, clipped on the exact cover-slide scale of −3 to +3."""
    stops = [-Z_LIMIT, 0, Z_LIMIT]
    return np.rint(np.stack([np.interp(z, stops, c) for c in COLOURS.T], axis=-1)).astype(np.uint8)


def raster(columns: np.ndarray, z: np.ndarray, show: np.ndarray | None = None) -> np.ndarray:
    """Draw reference z on the registered slice; dim (do not recolour) hidden voxels."""
    covered = columns >= 0
    voxels = columns[covered]
    rgb = np.empty((*columns.shape, 3), dtype=np.uint8)
    rgb[:] = BACKGROUND
    values = colours(z[voxels])
    if show is not None:
        values[~show[voxels]] = DIMMED
    rgb[covered] = values
    return rgb
