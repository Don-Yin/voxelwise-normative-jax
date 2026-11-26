"""Partition the original IXI flags by |z| and reconcile with experiment 010."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.batch_job import require_hugging_face_wrapper
from shared.voxelwisenormative.reference_bootstrap import instability, worker
from shared.voxelwisenormative.reference_grid import Z_THRESHOLD, Cell

logger = logging.getLogger(__name__)
BIN_LOWER_EDGES = (Z_THRESHOLD, 2.5, 3.1)
# exp_010 at 450 references: docs/figures/data/flag-stability.csv.
EXPECTED_UNSTABLE, EXPECTED_BASELINE = 200_530, 927_451


def summarize_bins(baseline: np.ndarray, original_abs_flags: np.ndarray, recurrence: np.ndarray) -> list[dict]:
    """Count unstable baseline flags and per-person fractions within each original-magnitude bin."""
    n_flags = int(baseline.sum())
    if (baseline.dtype != np.bool_ or original_abs_flags.shape != (n_flags,)
            or not np.isfinite(original_abs_flags).all() or np.any(original_abs_flags <= Z_THRESHOLD)):
        raise ValueError(f"expected {n_flags} finite baseline |z| values above {Z_THRESHOLD}, "
                         f"got magnitudes shape {original_abs_flags.shape} and baseline dtype {baseline.dtype}")
    classes = np.searchsorted(BIN_LOWER_EDGES, original_abs_flags, side="right") - 1
    selected = np.zeros_like(baseline)
    rows = []
    for i, lower in enumerate(BIN_LOWER_EDGES):
        upper = BIN_LOWER_EDGES[i + 1] if i + 1 < len(BIN_LOWER_EDGES) else None
        selected[baseline] = classes == i
        if not selected.any():
            raise ValueError(f"no original flags in |z| bin starting at {lower:g}; instability fraction is undefined")
        rows.append({"abs_z_lower": lower, "abs_z_upper_exclusive": upper, **instability(selected, recurrence)})
    return rows


def reconcile(rows: list[dict]) -> None:
    """Refuse a receipt unless both totals exactly reproduce the exp_010 result."""
    unstable = sum(row["unstable_flags"] for row in rows)
    flags = sum(row["baseline_flags"] for row in rows)
    if (unstable, flags) != (EXPECTED_UNSTABLE, EXPECTED_BASELINE):
        raise ValueError(f"IXI bins sum to {unstable:,} unstable / {flags:,} baseline flags; expected "
                         f"{EXPECTED_UNSTABLE:,} / {EXPECTED_BASELINE:,} from exp_010")
    logger.info("reconciled all bins: %d unstable / %d baseline flags", unstable, flags)


def summarize_ixi(baseline: np.ndarray, original_abs_flags: np.ndarray, recurrence: np.ndarray) -> list[dict]:
    """Validate the complete bin summary before returning it to the shared writer."""
    rows = summarize_bins(baseline, original_abs_flags, recurrence)
    reconcile(rows)
    return rows


def run(output: Path, *, smoke: bool) -> Path:
    """Run the real split only on Hugging Face, or exercise the worker on synthetic CPU data."""
    if not smoke:
        require_hugging_face_wrapper()
    if output.exists():
        raise FileExistsError(f"result already exists: {output}")
    cell = Cell("synthetic", 25) if smoke else Cell("ixi", 450)
    return worker(cell, 0, output, smoke=smoke, summarize_magnitudes=None if smoke else summarize_ixi)
