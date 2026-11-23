"""Run the pinned bootstrap and comparable site variants; keep scores ephemeral."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.basis import SITES
from shared.voxelwisenormative.batched_blr import N_ITER, Design, make_fitter
from shared.voxelwisenormative.bootstrap_score import Score
from shared.voxelwisenormative.bootstrap_score import cuda_scorer as a100_scorer
from shared.voxelwisenormative.bootstrap_score import scores_for as score_selection
from shared.voxelwisenormative.ixi_folds import row_digest

from .clusters import ClusterGrid, bootstrap_ranking, contrast_ranking, ranked
from .inputs import (
    MIN_HITS,
    N_FOLDS,
    N_REFITS,
    N_TRAIN,
    SEED,
    THRESHOLD,
    Cohort,
    draws,
    read_ixi,
    synthetic,
)
from .publish import Outputs
from .reference_size import smaller_reference, subset_report
from .z_summary import ZSummary, validate_cutoffs

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SiteFit:
    """One frozen covariate design and its reference-only selection of rows."""

    name: str
    train: np.ndarray
    include_site: bool
    control: str | None


def site_fits(data: Cohort) -> tuple[SiteFit, ...]:
    """Hold age/sex design fixed, and remove all site columns in the deletion contrasts."""
    all_rows = np.arange(len(data.train_rows))
    return (
        SiteFit("full-site", all_rows, True, None),
        SiteFit("no-site", all_rows, False, "full-site"),
        *(SiteFit(f"without-{site.lower()}", all_rows[data.reference_sites != site], False, "no-site")
          for site in ("Guys", "HH", "IOP")),
    )


def validate(data: Cohort) -> None:
    """Ensure interpretation of the pinned trailing site columns is unambiguous."""
    n = int(data.mask.sum())
    counts = len(data.train_rows), len(data.test_rows)
    if (data.train.shape != (counts[0], n) or data.test.shape != (counts[1], n)
            or data.phi_train.shape != (counts[0], 12) or data.phi_test.shape != (counts[1], 12)
            or np.intersect1d(data.train_rows, data.test_rows).size):
        raise ValueError(f"inconsistent split/design/mask: {data.train.shape=}, {data.test.shape=}, {counts=}, {n=}")
    categories = data.reference_sites[:, None] == np.asarray(SITES)
    if set(data.reference_sites) != set(SITES) or not np.array_equal(data.phi_train[:, -len(SITES):], categories):
        raise ValueError("expected the frozen design to end with three matching one-hot site columns")
    test_categories = data.phi_test[:, -len(SITES):]
    if not (np.isin(test_categories, (0, 1)).all() and (test_categories.sum(axis=1) == 1).all()):
        raise ValueError("invalid site indicator columns in the held-out design")


def scores_for(data: Cohort, rows: np.ndarray, scorer: Score, *, include_site: bool) -> np.ndarray:
    """Scale on selected training rows only; predict every held-out scan with one fit; the signed z of each scan and voxel."""
    stop = data.phi_train.shape[1] if include_site else data.phi_train.shape[1] - len(SITES)
    return score_selection(data.phi_train[rows, :stop], data.train[rows], data.phi_test[:, :stop], data.test, scorer)


def flags_for(data: Cohort, rows: np.ndarray, scorer: Score, *, include_site: bool) -> np.ndarray:
    """The scores of one fit whose absolute value exceeds THRESHOLD."""
    return np.abs(scores_for(data, rows, scorer, include_site=include_site)) > THRESHOLD


def cuda_scorer(chunk: int) -> tuple[Score, dict[str, str | None]]:
    """Use the common A100 scorer with this experiment's standard fitter."""
    return a100_scorer(chunk, fit_factory=make_fitter)


def fixture_score(rotated: Design, y: np.ndarray, yt: np.ndarray) -> np.ndarray:
    """Toy regression for local integration tests, not a substitute for the warped CUDA fit."""
    w = (rotated.G @ y) / (rotated.lam[:, None] + 2.0)
    residual_scale = np.maximum(np.std(y - rotated.G.T @ w, axis=0), 0.35)
    return (yt - rotated.H @ w) / residual_scale


def open_inputs(chunk: int, use_fixture: bool, fold: int = 0) -> tuple[Score, dict[str, str | None], Cohort]:
    """The scorer, its runtime record and the validated cohort; the real branch needs a CUDA Slurm or Hugging Face job."""
    runtime: dict[str, str | None]
    if use_fixture:
        scorer, data = fixture_score, synthetic(fold)
        runtime = {"device": "local toy regression"}
    else:
        scorer, runtime = cuda_scorer(chunk)
        data = read_ixi(fold)
    validate(data)
    return scorer, runtime, data


def bootstrap_refits(data: Cohort, scorer: Score, samples: np.ndarray, keep: Callable[[int, np.ndarray], None],
                     observe: Callable[[int, np.ndarray], None] | None = None) -> tuple[np.ndarray, list[float]]:
    """Fit every fixed draw with the site covariates; `keep` sees each refit's flags, `observe` (if given) its |z|.

    Returns the counts and the seconds of each fit; `observe` runs after the clock stops.
    """
    counts = np.zeros(data.test.shape, dtype=np.uint8)
    seconds = []
    for i, rows in enumerate(samples):
        begin = time.perf_counter()
        abs_z = np.abs(scores_for(data, rows, scorer, include_site=True))
        flagged = abs_z > THRESHOLD
        counts += flagged
        keep(i, flagged)
        seconds.append(time.perf_counter() - begin)
        if observe is not None:
            observe(i, abs_z)
        logger.info("[%d/%d] refit %.2fs, %d flags across the held-out scans",
                    i + 1, len(samples), seconds[-1], flagged.sum())
    return counts, seconds


def reference_subset(pool: Cohort, reference_n: int, fold: int) -> Cohort:
    """The age-stratified `reference_n` of the references (fold 0 only), validated like the full cohort."""
    if fold:
        raise ValueError(f"reference_n {reference_n} needs fold 0, got fold {fold}")
    data = smaller_reference(pool, reference_n)
    validate(data)
    logger.info("reference subset: %d of %d references, ages %.1f to %.1f", len(data.train_rows),
                len(pool.train_rows), data.reference_ages.min(), data.reference_ages.max())
    return data


def fit_once(data: Cohort, scorer: Score) -> tuple[np.ndarray, float]:
    """The fit on all references without resampling: its signed z for every held-out scan, and its seconds."""
    begin = time.perf_counter()
    z = scores_for(data, np.arange(len(data.train_rows)), scorer, include_site=True)
    seconds = time.perf_counter() - begin
    logger.info("fit without resampling: %.2fs", seconds)
    return z, seconds


def run_per_refit(results: Path, name: str, chunk: int, *, use_fixture: bool, fold: int = 0,
                  reference_n: int | None = None, z_summary: bool = False, cutoffs: tuple[float, ...] = ()) -> None:
    """Fit only the 64 draws and keep every refit's flags (bit-packed) for all held-out scans of one fold.

    Fold 0 (the default) is the frozen split and must reproduce exp_005's draws; folds 1-4 hold out a quarter of the
    frozen reference each (inputs.fold_split), with draws of the same seed over their own reference rows.
    `reference_n` (fold 0 only) keeps an age-stratified subset of the references, fits it once without resampling and
    draws the 64 samples from it; at 450 the draws are the original ones.
    `z_summary` also fits the references once without resampling, before the refits, and keeps that fit's z with the
    smallest and largest |z| of the refits in z-summary.npz (z_summary.py); without it nothing here differs.
    `cutoffs` (with z_summary) also keep every refit's flags and the single fit's at each stricter cutoff in cutoff-flags.npz.
    """
    started = time.perf_counter()
    cutoffs = validate_cutoffs(cutoffs, z_summary)  # before the first fit
    output = Outputs(results, name, refits_only=True, z_summary=z_summary, cutoff_flags=bool(cutoffs))
    output.require_new()
    scorer, runtime, data = open_inputs(chunk, use_fixture, fold)
    pool, extra, summary = data, {}, None
    if reference_n is not None:
        data = reference_subset(pool, reference_n, fold)
    if reference_n is not None or z_summary:
        original_z, seconds = fit_once(data, scorer)
        if reference_n is not None:
            extra = {"original_flags_packed": np.packbits(np.abs(original_z) > THRESHOLD, axis=-1, bitorder="little"),
                     "reference_cohort_rows_zero_based": data.train_rows.astype(np.int32)}
        if z_summary:
            summary = ZSummary(original_z, seconds, cutoffs)
    samples, draw_digest = draws(data, verify_original=not use_fixture and fold == 0
                                 and len(data.train_rows) == N_TRAIN)
    n_scans, n_voxels = data.test.shape
    packed = np.empty((len(samples), n_scans, (n_voxels + 7) // 8), dtype=np.uint8)

    def keep(i: int, flagged: np.ndarray) -> None:
        packed[i] = np.packbits(flagged, axis=-1, bitorder="little")

    def observe(_: int, abs_z: np.ndarray) -> None:
        if summary is not None:
            summary.add(abs_z)

    counts, refit_seconds = bootstrap_refits(data, scorer, samples, keep, observe if summary is not None else None)
    report = {"mode": "synthetic-refits" if use_fixture else "ixi-cuda-refits", "runtime": runtime,
              "held_out_scans": n_scans, "voxel_columns": n_voxels, "references": len(data.train_rows),
              "bootstrap_refits": len(samples), "bootstrap_seed": SEED, "bootstrap_cohort_rows_sha256": draw_digest,
              "optimizer_iterations": None if use_fixture else N_ITER, "chunk": chunk,
              "flag_rule": f"abs(z) > {THRESHOLD:g}", "design": "age, sex, site; the same refits as the main run",
              "fold": fold, "folds": N_FOLDS, "reference_cohort_rows_sha256": row_digest(data.train_rows),
              "mask_column_order": "C-order true voxels", "packed_axis": -1, "packed_bitorder": "little",
              "refit_seconds": refit_seconds, "seconds_before_export": time.perf_counter() - started}
    if reference_n is not None:
        report.update(subset_report(pool, data, samples))
    z_arrays = cutoff_arrays = None
    if summary is not None:
        report["z_summary"] = summary.receipt(counts, packed)
        z_arrays = summary.arrays(data.test_rows)
        cutoff_arrays = summary.cutoff_arrays(data.test_rows) if cutoffs else None
    output.publish_refits(packed, counts, data.test_rows, report, z_arrays, cutoff_arrays, **extra)


def run(results: Path, name: str, chunk: int, *, use_fixture: bool) -> None:
    """Fit 64 draws plus five site designs, rank spatial flips, and publish only aggregates."""
    started = time.perf_counter()
    output = Outputs(results, name)
    output.require_new()
    scorer, runtime, data = open_inputs(chunk, use_fixture)
    samples, draw_digest = draws(data, verify_original=not use_fixture)
    grid = ClusterGrid(data.mask, data.axial_indices)
    first = np.empty((3, *data.test.shape), dtype=bool)
    logger.info("%d held-out scans × %d voxels; %d fixed bootstrap draws", *data.test.shape, len(samples))

    def keep(i: int, flagged: np.ndarray) -> None:
        if i < len(first):
            first[i] = flagged

    counts, refit_seconds = bootstrap_refits(data, scorer, samples, keep)
    boot_rank = bootstrap_ranking(counts, first[0], grid)

    site_maps = {}
    site_info = []
    scanner_rows = []
    cases = site_fits(data)
    for i, case in enumerate(cases, start=1):
        begin = time.perf_counter()
        site_maps[case.name] = flags_for(data, case.train, scorer, include_site=case.include_site)
        site_info.append({"name": case.name, "reference_people": len(case.train), "include_site": case.include_site,
                          "control": case.control, "fit_seconds": time.perf_counter() - begin})
        logger.info("[%d/%d] %s, %d references, %.2fs", i, len(cases), case.name,
                    len(case.train), site_info[-1]["fit_seconds"])
        if case.control is not None:
            scanner_rows.extend(contrast_ranking(case.name, case.control, site_maps[case.name],
                                                 site_maps[case.control], grid))
    scan_rank = ranked(scanner_rows, "largest_flip_cluster_voxels", "flipped_voxels", "rank")
    report = {"mode": "synthetic" if use_fixture else "ixi-cuda", "runtime": runtime,
              "held_out_scans": len(data.test_rows), "voxel_columns": grid.n, "references": len(data.train_rows),
              "bootstrap_refits": N_REFITS, "bootstrap_seed": SEED, "bootstrap_cohort_rows_sha256": draw_digest,
              "optimizer_iterations": None if use_fixture else N_ITER, "chunk": chunk,
              "flag_rule": f"abs(z) > {THRESHOLD:g}", "recurring_count_minimum": MIN_HITS,
              "component_connectivity": 6, "mask_column_order": "C-order true voxels",
              "axial_mm": data.axial_mm, "axial_indices_zero_based": data.axial_indices,
              "refit_seconds": refit_seconds, "site_conditions": site_info,
              "seconds_before_export": time.perf_counter() - started}
    output.publish(counts, first, site_maps, data.test_rows, boot_rank, scan_rank, report)
    logger.info("largest fragile cluster: %s voxels in held-out row %s",
                boot_rank[0]["largest_cluster_voxels"], boot_rank[0]["held_out_row_zero_based"])
    logger.info("largest scanner contrast cluster: %s voxels, %s, held-out row %s",
                scan_rank[0]["largest_flip_cluster_voxels"], scan_rank[0]["contrast"], scan_rank[0]["held_out_row_zero_based"])
