"""--z-summary: what to keep from the refits so that any |z| threshold can be applied afterwards.

Besides the flags at |z| > THRESHOLD, the run keeps for every held-out scan and voxel, as fitted (float64):

    original_z       the signed z of the fit on all references, without resampling
    refit_min_abs_z  the smallest |z| over the refits
    refit_max_abs_z  the largest |z| over the refits

A voxel is flagged in every refit at threshold T exactly when its smallest |z| exceeds T, so "kept in all refits" at T is
`(|original_z| > T) & (refit_min_abs_z > T)`.  At T = THRESHOLD that is the run's own `recurrence_counts == refits`, voxel
by voxel; `ZSummary.check` stops the run when the two differ.  float32 would round a value just above T onto the float32
threshold and break that equality for a voxel now and then, which is why the arrays stay float64.

--refit-flag-cutoffs T,T,... adds the flags themselves at stricter cutoffs, for what needs each refit's own flags (one
refit's column of a page, cluster-forming cutoffs).  cutoff-flags.npz holds, per cutoff, the flags `abs(z) > T` of every refit
and of the single fit, packed like refit_flags_packed.  Per cutoff the AND of the refits' flags must be `refit_min_abs_z > T`
and their OR `refit_max_abs_z > T`, and at THRESHOLD the same code must give back the run's own refit flags (not stored);
`ZSummary.check_cutoffs` stops the run otherwise, before anything is written.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .inputs import N_REFITS, THRESHOLD

FILE = "z-summary.npz"
CUTOFF_FILE = "cutoff-flags.npz"
MAX_CUTOFFS = 10
ARRAYS = {"original_z": "signed z of the fit on all references, no resampling",
          "refit_min_abs_z": "smallest |z| over the refits", "refit_max_abs_z": "largest |z| over the refits"}
CUTOFF_ARRAYS = {"cutoffs": "float64 (cutoffs,): the cutoffs T as given",
                 "refit_flags_packed": "uint8 (cutoffs, refits, scans, ceil(voxels / 8)): abs(z) > T of every refit, packed like refit-flags.npz",
                 "original_flags_packed": "uint8 (cutoffs, scans, ceil(voxels / 8)): abs(original_z) > T of the fit without resampling"}


def validate_cutoffs(cutoffs: Iterable[float], z_summary: bool = True) -> tuple[float, ...]:
    """The cutoffs as floats: none, or with z_summary 1 to MAX_CUTOFFS distinct finite values above THRESHOLD (refit-flags.npz has it)."""
    values = tuple(float(cutoff) for cutoff in cutoffs)
    if values and not z_summary:
        raise ValueError("cutoffs need z_summary")
    if len(values) > MAX_CUTOFFS or len(set(values)) != len(values) or not all(np.isfinite(v) and v > THRESHOLD for v in values):
        raise ValueError(f"cutoffs must be at most {MAX_CUTOFFS} distinct finite values above {THRESHOLD:g}, got {values}")
    return values


def pack(flags: np.ndarray) -> np.ndarray:
    """Flags packed along the voxel axis, bit order little, as refit_flags_packed is."""
    return np.packbits(flags, axis=-1, bitorder="little")


class ZSummary:
    """The single fit's z and the smallest and largest |z| over the refits added so far; with cutoffs, each refit's flags too."""

    def __init__(self, original_z: np.ndarray, fit_seconds: float, cutoffs: tuple[float, ...] = (), refits: int = N_REFITS) -> None:
        if original_z.dtype != np.float64 or original_z.ndim != 2 or not np.isfinite(original_z).all():
            raise ValueError(f"the single fit's z must be a finite float64 (scans, voxels) array, got {original_z.dtype} {original_z.shape}")
        self.original_z = original_z
        self.fit_seconds = fit_seconds
        self.refits = 0
        self.smallest = np.full(original_z.shape, np.inf)
        self.largest = np.full(original_z.shape, -np.inf)
        self.cutoffs = np.array(validate_cutoffs(cutoffs), dtype=np.float64)
        self.flags: np.ndarray | None = None  # (cutoffs + 1, refits, scans, bytes): the last level is THRESHOLD, checked and dropped
        scans, voxels = original_z.shape
        self.original_flags = np.empty((0, scans, (voxels + 7) // 8), dtype=np.uint8)
        if len(self.cutoffs):
            self.flags = np.zeros((len(self.cutoffs) + 1, refits, scans, (voxels + 7) // 8), dtype=np.uint8)
            self.original_flags = np.stack([pack(np.abs(original_z) > cutoff) for cutoff in self.cutoffs])

    def add(self, abs_z: np.ndarray) -> None:
        """Fold one refit's |z| in."""
        if abs_z.shape != self.original_z.shape or abs_z.dtype != np.float64:
            raise ValueError(f"a refit's |z| must be float64 of shape {self.original_z.shape}, got {abs_z.dtype} {abs_z.shape}")
        if self.flags is not None:
            if self.refits >= self.flags.shape[1]:
                raise ValueError(f"more than the {self.flags.shape[1]} refits that were announced")
            for level, cutoff in enumerate((*self.cutoffs, THRESHOLD)):
                self.flags[level, self.refits] = pack(abs_z > cutoff)
        np.minimum(self.smallest, abs_z, out=self.smallest)
        np.maximum(self.largest, abs_z, out=self.largest)
        self.refits += 1

    def check(self, counts: np.ndarray) -> dict[str, float | int]:
        """At THRESHOLD, 'smallest |z| above it' must be 'flagged in every refit' for every voxel; returns what was compared."""
        if not self.refits or counts.shape != self.smallest.shape:
            raise ValueError(f"{self.refits} refits added; recurrence counts {counts.shape} against z {self.smallest.shape}")
        above = self.smallest > THRESHOLD
        every = counts == self.refits
        differ = int((above != every).sum())
        if differ:
            raise ValueError(f"refit_min_abs_z > {THRESHOLD:g} and recurrence_counts == {self.refits} differ in {differ:,} of "
                             f"{counts.size:,} entries; the z summary is not the one the flags came from")
        return {"threshold": THRESHOLD, "entries": int(counts.size), "above_threshold_in_every_refit": int(every.sum()),
                "entries_that_differ": differ}

    def check_cutoffs(self, refit_flags: np.ndarray) -> dict[str, object]:
        """Per cutoff the refits' flags must AND to `smallest > T` and OR to `largest > T`; at THRESHOLD they must be `refit_flags`."""
        if self.flags is None or self.refits != self.flags.shape[1]:
            raise ValueError(f"{self.refits} refits added, no cutoffs or not all refits to check")
        for level, cutoff in enumerate((*(float(c) for c in self.cutoffs), THRESHOLD)):
            for name, kept, bound in (("AND", np.bitwise_and.reduce(self.flags[level], axis=0), self.smallest),
                                      ("OR", np.bitwise_or.reduce(self.flags[level], axis=0), self.largest)):
                differ = int(np.unpackbits(kept ^ pack(bound > cutoff)).sum())
                if differ:
                    raise ValueError(f"at cutoff {cutoff:g} the {name} of the refits' flags differs from "
                                     f"refit_{'min' if name == 'AND' else 'max'}_abs_z > {cutoff:g} in {differ:,} entries")
        if self.flags[-1].shape != refit_flags.shape or not np.array_equal(self.flags[-1], refit_flags):
            raise ValueError(f"the flags at {THRESHOLD:g} that the cutoff code makes differ from refit_flags_packed "
                             f"in {int(np.unpackbits(self.flags[-1] ^ refit_flags).sum()):,} entries")
        return {"cutoffs": [float(c) for c in self.cutoffs], "refits": self.refits, "entries_per_cutoff": int(self.smallest.size),
                "checked": ["AND of the refits' flags == refit_min_abs_z > cutoff, OR == refit_max_abs_z > cutoff",
                            f"flags at {THRESHOLD:g} made by the same code == refit_flags_packed (not stored)"],
                "entries_that_differ": 0}

    def receipt(self, counts: np.ndarray, refit_flags: np.ndarray | None = None) -> dict[str, object]:
        """The receipt's `z_summary` entry: the file, what is in it, the single fit's seconds, the result of `check`, and with cutoffs theirs."""
        entry: dict[str, object] = {"file": FILE, "arrays": ARRAYS, "dtype": "float64", "original_fit_seconds": self.fit_seconds,
                                    "check": self.check(counts)}
        if self.flags is not None:
            if refit_flags is None:
                raise ValueError("the refits' own flags are needed to check the cutoffs")
            entry["cutoff_flags"] = {"file": CUTOFF_FILE, "cutoffs": [float(c) for c in self.cutoffs], "arrays": CUTOFF_ARRAYS,
                                     "check": self.check_cutoffs(refit_flags)}
        return entry

    def arrays(self, held_out_rows: np.ndarray) -> dict[str, np.ndarray]:
        """What goes into FILE: the three arrays and the cohort rows of the scans they belong to."""
        return {"original_z": self.original_z, "refit_min_abs_z": self.smallest, "refit_max_abs_z": self.largest,
                "held_out_cohort_rows_zero_based": held_out_rows.astype(np.int32, copy=False)}

    def cutoff_arrays(self, held_out_rows: np.ndarray) -> dict[str, np.ndarray]:
        """What goes into CUTOFF_FILE: the cutoffs, every refit's flags and the single fit's at each, and the cohort rows."""
        if self.flags is None:
            raise ValueError("no cutoffs were given")
        return {"cutoffs": self.cutoffs, "refit_flags_packed": self.flags[:-1], "original_flags_packed": self.original_flags,
                "held_out_cohort_rows_zero_based": held_out_rows.astype(np.int32, copy=False)}
