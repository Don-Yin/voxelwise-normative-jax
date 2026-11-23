"""Commit the transfer allowlist and publish a checksum receipt only after writing it."""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shared.voxelwisenormative.artifacts import sha256

from .clusters import Row
from .z_summary import CUTOFF_FILE
from .z_summary import FILE as Z_SUMMARY_FILE

logger = logging.getLogger(__name__)


def table(path: Path, rows: list[Row]) -> None:
    """Write a ranked TSV with a stable header and no participant identifiers."""
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@dataclass(frozen=True)
class Outputs:
    """New run directories only; the result receipt is the completion marker."""

    results: Path
    name: str
    refits_only: bool = False
    z_summary: bool = False  # per-refit runs only: also keep z-summary.npz (z_summary.py)
    cutoff_flags: bool = False  # with z_summary: also keep cutoff-flags.npz

    @property
    def stage(self) -> Path:
        """Incomplete output is private to this experiment."""
        return self.results.parent / ".intermediates" / self.name

    @property
    def destinations(self) -> dict[str, Path]:
        """Map approved export filenames to their purpose-specific output paths."""
        if self.refits_only:
            files = {"refit-flags.npz": self.results / "arrays" / self.name / "refit-flags.npz"}
            if self.z_summary:
                files[Z_SUMMARY_FILE] = self.results / "arrays" / self.name / Z_SUMMARY_FILE
            if self.cutoff_flags:
                files[CUTOFF_FILE] = self.results / "arrays" / self.name / CUTOFF_FILE
            return files
        return {"flags.npz": self.results / "arrays" / self.name / "flags.npz",
                "bootstrap-ranking.tsv": self.results / "tables" / self.name / "bootstrap-ranking.tsv",
                "scanner-ranking.tsv": self.results / "tables" / self.name / "scanner-ranking.tsv"}

    @property
    def receipt(self) -> Path:
        """Validation and provenance for the completed run, held on the cluster."""
        return self.results / "json" / self.name / "result.json"

    def require_new(self) -> None:
        """Reject reuse of a run name before computing any fits."""
        for path in {self.stage, self.receipt.parent, *(p.parent for p in self.destinations.values())}:
            if path.exists():
                raise FileExistsError(f"refusing to overwrite run {self.name}: {path}")

    def publish(self, counts, first, site_maps, test_rows, bootstrap, scanner, report) -> None:
        """Store only flag counts, packed flags, the index, and ranked summaries."""
        self.require_new()
        self.stage.mkdir(parents=True, exist_ok=False)
        packed_first = np.packbits(first, axis=-1, bitorder="little")
        packed_sites = np.packbits(np.stack(tuple(site_maps.values())), axis=-1, bitorder="little")
        np.savez_compressed(self.stage / "flags.npz", recurrence_counts=counts,
                            first_three_flags_packed=packed_first, site_flags_packed=packed_sites,
                            site_conditions=np.asarray(tuple(site_maps)),
                            held_out_cohort_rows_zero_based=test_rows.astype(np.int32, copy=False))
        table(self.stage / "bootstrap-ranking.tsv", bootstrap)
        table(self.stage / "scanner-ranking.tsv", scanner)
        self.commit(report)

    def publish_refits(self, packed, counts, test_rows, report, z_arrays: dict[str, Any] | None = None,
                       cutoff_arrays: dict[str, Any] | None = None, **extra: np.ndarray) -> None:
        """Store every refit's packed flags for all held-out scans, with the recurrence counts as a self-check.

        `extra` arrays (only from --reference-n) are added to the same archive under their own keys; `z_arrays` (only from
        --z-summary) go uncompressed into their own file, which floating-point values hardly shrink; `cutoff_arrays` (only
        from --refit-flag-cutoffs) go compressed into theirs, since packed flags at a strict cutoff are mostly zero.
        """
        if self.z_summary != (z_arrays is not None):
            raise ValueError(f"z_summary is {self.z_summary} but z_arrays is {'given' if z_arrays is not None else 'missing'}")
        if self.cutoff_flags != (cutoff_arrays is not None) or (self.cutoff_flags and not self.z_summary):
            raise ValueError(f"cutoff_flags is {self.cutoff_flags} (it needs z_summary) but cutoff_arrays is "
                             f"{'given' if cutoff_arrays is not None else 'missing'}")
        self.require_new()
        self.stage.mkdir(parents=True, exist_ok=False)
        arrays: dict[str, Any] = {"refit_flags_packed": packed, "recurrence_counts": counts,
                                  "held_out_cohort_rows_zero_based": test_rows.astype(np.int32, copy=False), **extra}
        np.savez_compressed(self.stage / "refit-flags.npz", **arrays)
        if z_arrays is not None:
            np.savez(self.stage / Z_SUMMARY_FILE, **z_arrays)
        if cutoff_arrays is not None:
            np.savez_compressed(self.stage / CUTOFF_FILE, **cutoff_arrays)
        self.commit(report)

    def commit(self, report) -> None:
        """Checksum the staged files, then move them and the receipt into place; the receipt goes last."""
        final = self.destinations
        report.update(schema_version=1, run_name=self.name, packed_axis=-1, packed_bitorder="little",
                      outputs={name: {"path": str(path.relative_to(self.results)),
                                      "sha256": sha256(self.stage / name), "bytes": (self.stage / name).stat().st_size}
                               for name, path in final.items()})
        partial_receipt = self.stage / "result.json"
        partial_receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for directory in {p.parent for p in (*final.values(), self.receipt)}:
            directory.mkdir(parents=True, exist_ok=False)
        for name, path in final.items():
            (self.stage / name).rename(path)
        partial_receipt.rename(self.receipt)
        self.stage.rmdir()
        logger.info("saved completed run receipt to %s", self.receipt)
