"""Two runs' npz files, array by array and bit for bit (jobs/compare_arrays.py).

It shows that a change to the code (the --refit-flag-cutoffs option, a new clusters.py) left a fold's refit-flags.npz and
z-summary.npz as they were. Bytes are compared, not values: -0.0 differs from 0.0, and a NaN equals one with the same bits.
The files themselves are not compared: a zip holds the time it was written.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .inputs import THRESHOLD

FILES = ("refit-flags.npz", "z-summary.npz")
UNSIGNED = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
BITS_SET = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def bit_patterns(array: np.ndarray) -> np.ndarray:
    """The array's memory as unsigned integers of the same width."""
    if array.dtype.kind not in "biuf" or array.dtype.itemsize not in UNSIGNED:
        raise ValueError(f"cannot compare the bits of {array.dtype}")
    return np.ascontiguousarray(array).view(UNSIGNED[array.dtype.itemsize])


def compare_array(old: np.ndarray, new: np.ndarray, flags_packed: bool = False) -> dict[str, object]:
    """dtype, shape and element count; `differ` is None when dtype or shape differ (`mismatch` says how), else the number of
    elements whose bits differ, with where the first is; packed flags add the bits that differ, floats the largest difference
    and how many entries have a flag at THRESHOLD that moves."""
    row: dict[str, object] = {"dtype": str(old.dtype), "shape": tuple(int(n) for n in old.shape), "elements": int(old.size)}
    if old.dtype != new.dtype or old.shape != new.shape:
        return row | {"differ": None, "mismatch": f"baseline {old.dtype} {old.shape}, new {new.dtype} {new.shape}"}
    moved = bit_patterns(old) != bit_patterns(new)
    row["differ"] = int(np.count_nonzero(moved))
    if row["differ"]:
        row["first"] = tuple(int(i) for i in np.unravel_index(int(np.argmax(moved)), moved.shape))
        if flags_packed:
            row["flag_bits"] = int(BITS_SET[old ^ new].sum(dtype=np.int64))
        if old.dtype.kind == "f":
            row["largest_difference"] = float(np.max(np.abs(old[moved] - new[moved])))
            row["flags_changed"] = int(np.count_nonzero((np.abs(old) > THRESHOLD) != (np.abs(new) > THRESHOLD)))
    return row


def compare_runs(baseline: Path, new: Path, names: Sequence[str] = FILES) -> list[dict[str, object]]:
    """One row per array of each named file, keys in file order; both folders must hold every file."""
    missing = [str(folder / name) for folder in (baseline, new) for name in names if not (folder / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing: {', '.join(missing)}")
    rows: list[dict[str, object]] = []
    for name in names:
        with np.load(baseline / name, allow_pickle=False) as old, np.load(new / name, allow_pickle=False) as fresh:
            for key in dict.fromkeys([*old.files, *fresh.files]):
                row: dict[str, object] = {"file": name, "key": key}
                if key in old.files and key in fresh.files:
                    rows.append(row | compare_array(old[key], fresh[key], key.endswith("_packed")))
                else:
                    rows.append(row | {"differ": None, "mismatch": f"only in the {'baseline' if key in old.files else 'new'} file"})
    return rows


def identical(rows: Sequence[dict[str, object]]) -> bool:
    """True when every array matches bit for bit and none is missing."""
    return all(row["differ"] == 0 for row in rows)


def verdict(row: dict[str, object]) -> str:
    """'identical', or what differs, in one line."""
    if row.get("mismatch"):
        return f"DIFFERENT: {row['mismatch']}"
    if row["differ"] == 0:
        return f"identical ({row['elements']:,} elements)"
    parts = [f"DIFFERENT: {row['differ']:,} of {row['elements']:,} elements ({100 * row['differ'] / row['elements']:.3g}%), first at {row['first']}"]  # type: ignore[index, operator]
    if "flag_bits" in row:
        parts.append(f"{row['flag_bits']:,} flag bits")
    if "largest_difference" in row:
        parts.append(f"largest difference {row['largest_difference']:.3g}, entries whose flag at {THRESHOLD:g} changes: {row['flags_changed']:,}")
    return "; ".join(parts)


def render(rows: Sequence[dict[str, object]], baseline: str, new: str) -> list[str]:
    """A table of the rows with the two folders named, and the one-line total."""
    cells = [(str(row["file"]), str(row["key"]), f"{row.get('dtype', '-')} {row.get('shape', '')}", verdict(row)) for row in rows]
    widths = [max(len(cell[i]) for cell in cells) for i in range(3)]
    lines = [f"baseline: {baseline}", f"new:      {new}"]
    lines += ["  ".join(cell[i].ljust(widths[i]) for i in range(3)) + "  " + cell[3] for cell in cells]
    differing = sum(row["differ"] != 0 for row in rows)
    files = len({row["file"] for row in rows})
    lines.append(f"all {len(rows)} arrays of {files} files are bit-for-bit identical" if not differing
                 else f"{differing} of {len(rows)} arrays differ")
    return lines
