"""The fetched z-summary and cutoff-flags runs: receipts checked against the files, arrays cross-checked, one fold at a time."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.artifacts import sha256

from .inputs import N_FOLDS, N_REFITS, N_SCANS, N_VOXELS, THRESHOLD

logger = logging.getLogger(__name__)
RUN = "hugging-face-a100-z-summary-fold{fold}"
CUTOFF_RUN = "hugging-face-a100-cutoff-flags-fold{fold}"  # the same files plus cutoff-flags.npz: each refit's flags at stricter cutoffs
FILES = ("refit-flags.npz", "z-summary.npz")


@dataclass(frozen=True)
class FoldRun:
    """One fetched run: where its two arrays are, and the digests the receipt promised and the files kept."""

    fold: int
    name: str
    refit_flags: Path
    z_summary: Path
    receipt: dict
    files: dict[str, dict[str, str | int]]
    cutoff_flags: Path | None = None


@dataclass(frozen=True)
class FoldArrays:
    """One fold's held-out people: cohort rows, single-fit z and the packed flags of every refit."""

    rows: np.ndarray
    z: np.ndarray
    packed: np.ndarray


def open_fold(results: Path, fold: int, cutoffs: bool = False) -> FoldRun:
    """Find fold `fold` under .results/hugging-face/<run>/ and verify its arrays against the run's receipt.

    With `cutoffs` the run is the cutoff-flags one, whose cutoff-flags.npz is verified as well.
    """
    name = (CUTOFF_RUN if cutoffs else RUN).format(fold=fold)
    root = results / "hugging-face" / name
    try:
        receipt = json.loads((root / "json" / name / "result.json").read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the receipt of run {name}: {root / 'json' / name / 'result.json'}") from error
    expected = {"run_name": name, "fold": fold, "folds": N_FOLDS, "bootstrap_refits": N_REFITS, "voxel_columns": N_VOXELS,
                "flag_rule": f"abs(z) > {THRESHOLD:g}", "mask_column_order": "C-order true voxels", "packed_axis": -1,
                "packed_bitorder": "little"}
    wrong = {key: (receipt.get(key), value) for key, value in expected.items() if receipt.get(key) != value}
    if wrong or receipt["z_summary"]["check"]["entries_that_differ"]:
        raise ValueError(f"{root}: the receipt is not the expected run (found, expected): {wrong}; z_summary {receipt['z_summary']['check']}")
    files = {}
    for key in (*FILES, *(("cutoff-flags.npz",) if cutoffs else ())):
        record, path = receipt["outputs"][key], root / receipt["outputs"][key]["path"]
        found = (sha256(path), path.stat().st_size)
        if found != (record["sha256"], record["bytes"]):
            raise ValueError(f"{path} is not the file its receipt describes: found sha256 {found[0]}, {found[1]} bytes; receipt {record}")
        files[key] = {"path": str(path.relative_to(results)), "sha256": found[0], "bytes": found[1]}
    return FoldRun(fold, name, root / receipt["outputs"]["refit-flags.npz"]["path"], root / receipt["outputs"]["z-summary.npz"]["path"],
                   receipt, files, root / receipt["outputs"]["cutoff-flags.npz"]["path"] if cutoffs else None)


def read_fold(run: FoldRun) -> FoldArrays:
    """Load one fold and check that its arrays agree: same people, flags that sum to the counts, no stray padding bits."""
    with np.load(run.refit_flags, allow_pickle=False) as archive:
        packed, counts, rows = archive["refit_flags_packed"], archive["recurrence_counts"], archive["held_out_cohort_rows_zero_based"]
    with np.load(run.z_summary, allow_pickle=False) as archive:
        z, z_rows, smallest = archive["original_z"], archive["held_out_cohort_rows_zero_based"], archive["refit_min_abs_z"]
    people = run.receipt["held_out_scans"]
    shapes = {"refit_flags_packed": (packed.shape, (N_REFITS, people, (N_VOXELS + 7) // 8)),
              "recurrence_counts": (counts.shape, (people, N_VOXELS)), "original_z": (z.shape, (people, N_VOXELS)),
              "held_out_cohort_rows": (rows.shape, (people,))}
    bad = {key: found for key, (found, wanted) in shapes.items() if found != wanted}
    if bad or (packed.dtype, counts.dtype, z.dtype) != (np.uint8, np.uint8, np.float64) or not np.isfinite(z).all():
        raise ValueError(f"{run.name}: unexpected arrays {bad}; dtypes {packed.dtype} {counts.dtype} {z.dtype}; expected shapes {shapes}")
    if not np.array_equal(rows, z_rows) or not (np.diff(rows) > 0).all():
        raise ValueError(f"{run.name}: refit-flags.npz and z-summary.npz list other scans, or the rows are not increasing: {rows[:5]} against {z_rows[:5]}")
    total = np.zeros(counts.shape, dtype=np.uint8)
    for refit in packed:
        total += np.unpackbits(refit, axis=-1, count=N_VOXELS, bitorder="little")
    if not np.array_equal(total, counts) or (packed[..., -1] >> (N_VOXELS % 8)).any():
        raise ValueError(f"{run.name}: the packed refit flags do not sum to recurrence_counts, or the padding bits are set")
    if not np.array_equal(smallest > THRESHOLD, counts == N_REFITS):
        raise ValueError(f"{run.name}: refit_min_abs_z > {THRESHOLD:g} differs from recurrence_counts == {N_REFITS}")
    logger.info("fold %d: %d people, %d voxels in all %d refits, arrays agree", run.fold, people, int((counts == N_REFITS).sum()), N_REFITS)
    return FoldArrays(rows, z, packed)


def read_cutoff_fold(run: FoldRun, threshold: float) -> FoldArrays:
    """One fold at one of the cutoff run's stricter cutoffs: the single fit's signed z and every refit's flags at `threshold`.

    Checked against the z summary: the single fit's flags are `abs(original_z) > T`, the refits' flags AND to `refit_min_abs_z > T`
    and OR to `refit_max_abs_z > T`, and no padding bit is set.
    """
    if run.cutoff_flags is None:
        raise ValueError(f"{run.name} was opened without its cutoff-flags.npz")
    people, width = run.receipt["held_out_scans"], (N_VOXELS + 7) // 8
    with np.load(run.cutoff_flags, allow_pickle=False) as archive:
        cutoffs, rows = archive["cutoffs"], archive["held_out_cohort_rows_zero_based"]
        level = np.flatnonzero(cutoffs == threshold)
        if cutoffs.ndim != 1 or len(level) != 1:
            raise ValueError(f"{run.cutoff_flags}: the cutoffs {cutoffs.tolist()} do not hold {threshold:g} exactly once")
        refit_all, single_all = archive["refit_flags_packed"], archive["original_flags_packed"]
        shapes = {"refit_flags_packed": (refit_all.shape, (len(cutoffs), N_REFITS, people, width)),
                  "original_flags_packed": (single_all.shape, (len(cutoffs), people, width)), "held_out_cohort_rows": (rows.shape, (people,))}
        bad = {key: found for key, (found, wanted) in shapes.items() if found != wanted}
        if bad or refit_all.dtype != np.uint8 or single_all.dtype != np.uint8:
            raise ValueError(f"{run.cutoff_flags}: unexpected arrays {bad}; dtypes {refit_all.dtype} {single_all.dtype}; expected shapes {shapes}")
        flags, single = np.ascontiguousarray(refit_all[level[0]]), single_all[level[0]]
        del refit_all
    with np.load(run.z_summary, allow_pickle=False) as archive:
        z, z_rows = archive["original_z"], archive["held_out_cohort_rows_zero_based"]
        smallest, largest = archive["refit_min_abs_z"], archive["refit_max_abs_z"]
    if z.shape != (people, N_VOXELS) or not np.isfinite(z).all() or not np.array_equal(rows, z_rows) or not (np.diff(rows) > 0).all():
        raise ValueError(f"{run.name}: z-summary.npz has shape {z.shape} or lists other scans than cutoff-flags.npz, or the rows do not increase")

    def packed(values: np.ndarray) -> np.ndarray:
        return np.packbits(values, axis=-1, bitorder="little")

    wrong = {"the single fit's flags": not np.array_equal(single, packed(np.abs(z) > threshold)),
             "the AND of the refits' flags": not np.array_equal(np.bitwise_and.reduce(flags, axis=0), packed(smallest > threshold)),
             "the OR of the refits' flags": not np.array_equal(np.bitwise_or.reduce(flags, axis=0), packed(largest > threshold)),
             "the padding bits": bool((flags[..., -1] >> (N_VOXELS % 8)).any())}
    if any(wrong.values()):
        raise ValueError(f"{run.name} at {threshold:g}: " + ", ".join(key for key, bad_one in wrong.items() if bad_one) + " disagree with z-summary.npz")
    logger.info("fold %d at %g: %d people, single-fit flags and the AND and OR of the %d refits agree with the z summary", run.fold, threshold, people, N_REFITS)
    return FoldArrays(rows, z, flags)


def check_cohort(rows_by_fold: dict[int, np.ndarray]) -> None:
    """Every one of the 563 scans is held out in exactly one of the folds that were read, and no row is listed twice."""
    rows = np.concatenate(list(rows_by_fold.values()))
    if len(rows) != len(np.unique(rows)) or rows.min() < 0 or rows.max() >= N_SCANS:
        raise ValueError(f"cohort rows repeat or fall outside 0..{N_SCANS - 1}: {len(rows)} rows, {len(np.unique(rows))} distinct, "
                         f"folds {sorted(rows_by_fold)}")
    if len(rows_by_fold) == N_FOLDS and len(rows) != N_SCANS:
        raise ValueError(f"the {N_FOLDS} folds hold out {len(rows)} scans, not {N_SCANS}")
