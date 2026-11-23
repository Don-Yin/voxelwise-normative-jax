"""A smaller reference: an age-stratified subset of the frozen one, and the per-scan count of removed flags."""
from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np

from shared.voxelwisenormative.artifacts import sha256
from shared.voxelwisenormative.basis import SITES

from .inputs import MIN_HITS, N_REFITS, N_TRAIN, SEED, THRESHOLD, Cohort
from .publish import table

logger = logging.getLogger(__name__)

MIN_ROW_CORRELATION = 0.95  # of per-scan flag counts, the one fit's against the 64 refits'; row-shuffled scans give about 0
SUBSET_RULE = ("the youngest and the oldest reference, then one at random from each of N - 2 age strata of the others "
               "(ages sorted, ties by cohort row; stratum k is sorted positions k*M//(N-2) to (k+1)*M//(N-2), M others); "
               "rows kept in reference order")


def age_stratified(ages: np.ndarray, n: int) -> np.ndarray:
    """Sorted positions of n references; n equal to the pool returns every position in order."""
    if ages.ndim != 1 or not np.isfinite(ages).all() or not 3 <= n <= ages.size:
        raise ValueError(f"need finite ages and 3 <= n <= {ages.size} references, got {n=}, {ages.shape=}")
    order = np.argsort(ages, kind="stable")
    inner = order[1:-1]
    edges = np.arange(n - 1) * inner.size // (n - 2)  # sizes differ by at most one and alternate, so no age drifts
    picks = inner[np.random.default_rng((SEED, n)).integers(edges[:-1], edges[1:])]
    return np.sort(np.r_[order[0], order[-1], picks])


def smaller_reference(data: Cohort, n: int) -> Cohort:
    """The cohort with only the age-stratified n references; held-out scans, mask and basis unchanged."""
    keep = age_stratified(data.reference_ages, n)
    return replace(data, train=data.train[keep], phi_train=data.phi_train[keep], train_rows=data.train_rows[keep],
                   reference_sites=data.reference_sites[keep], reference_ages=data.reference_ages[keep])


def subset_report(pool: Cohort, subset: Cohort, samples: np.ndarray) -> dict:
    """Receipt fields: rule, seed, age range and site mix of the subset, and the fewest people per site in a draw."""
    n = len(subset.train_rows)
    drawn = subset.reference_sites[samples]
    return {"reference_n": n, "reference_pool": len(pool.train_rows), "reference_subset_rule": SUBSET_RULE,
            "reference_subset_seed": [SEED, n],
            "reference_age_range": [float(subset.reference_ages.min()), float(subset.reference_ages.max())],
            "reference_pool_age_range": [float(pool.reference_ages.min()), float(pool.reference_ages.max())],
            "reference_age_quartiles": np.percentile(subset.reference_ages, [25, 50, 75]).tolist(),
            "reference_site_counts": {s: int((subset.reference_sites == s).sum()) for s in SITES},
            "draw_site_minimum": {s: int((drawn == s).sum(axis=1).min()) for s in SITES},
            "design": "age, sex, site; the frozen basis; original fit and 64 draws of the N-person reference",
            "original_fit": "one fit on all N references, no resampling: original_flags_packed"}


def removed_ranking(original: np.ndarray, counts: np.ndarray, cohort_rows: np.ndarray,
                    on_planes: np.ndarray) -> list[dict]:
    """Rank held-out scans by original flags that recur in fewer than MIN_HITS refits, then by lower row."""
    if original.shape != counts.shape or original.dtype != bool or on_planes.shape != original.shape[1:]:
        raise ValueError(f"mismatched inputs: {original.shape=} {original.dtype=}, {counts.shape=}, {on_planes.shape=}")
    removed = original & (counts < MIN_HITS)
    rows = [{"held_out_row_zero_based": i, "held_out_cohort_row_zero_based": int(cohort_rows[i]),
             "original_flags": int(original[i].sum()), "removed_voxels": int(removed[i].sum()),
             "removed_share": round(float(removed[i].sum() / max(original[i].sum(), 1)), 4),
             "removed_on_planes": int((removed[i] & on_planes).sum())} for i in range(len(original))]
    rows.sort(key=lambda r: (-r["removed_voxels"], r["held_out_row_zero_based"]))
    return [{"rank": k, **r} for k, r in enumerate(rows, start=1)]


def original_flags(path: Path, counts: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """abs(z) > THRESHOLD of the one fit on all references: exp_003's z_test.npy, or the original_z of a z-summary.npz.

    A z-summary.npz names its scans, which must be the archive's. A z_test.npy does not, so its rows are accepted only if each
    scan's flag count follows the archive's own (the counts of 64 refits); a row mix-up would break that at once.
    """
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            z, z_rows = archive["original_z"], archive["held_out_cohort_rows_zero_based"]
        if not np.array_equal(z_rows, rows):
            raise ValueError(f"{path} holds other scans than the archive: {z_rows[:3]}... against {rows[:3]}...")
    else:
        z = np.load(path, mmap_mode="r", allow_pickle=False)
    if z.shape != counts.shape:
        raise ValueError(f"{path} has shape {z.shape}, the archive's counts {counts.shape}")
    flags = np.abs(np.asarray(z)) > THRESHOLD
    follows = np.corrcoef(flags.sum(axis=1), counts.sum(axis=1, dtype=np.int64))[0, 1]
    if not follows > MIN_ROW_CORRELATION:
        raise ValueError(f"{path}: the flag count per scan follows the refits' with correlation {follows:.3f}; are the rows the same scans?")
    return flags


def rank_fetched(folder: Path, on_planes: np.ndarray, original_z: Path | None = None) -> Path:
    """Check a fetched per-refit archive against its receipt, then write removed-ranking.tsv beside it.

    A --reference-n run keeps its own `original_flags_packed`. A full-size run has none,
    so `original_z` names the z of its one fit, as `original_flags` reads it.
    """
    receipt = json.loads((folder / "result.json").read_text())
    archive, record = folder / "refit-flags.npz", receipt["outputs"]["refit-flags.npz"]
    if (sha256(archive), archive.stat().st_size) != (record["sha256"], record["bytes"]):
        raise ValueError(f"{archive} is not the output its receipt describes: {record}")
    subset = "reference_n" in receipt
    if subset == (original_z is not None):
        raise ValueError(f"{folder}: " + ("a --reference-n run has its own original flags, so --original-z does not apply" if subset else
                                          "this run has no original_flags_packed (it is no --reference-n run): give --original-z"))
    if original_z is not None and original_z.suffix != ".npz" and (receipt.get("fold", 0), receipt["references"]) != (0, N_TRAIN):
        raise ValueError(f"{original_z.name} is the fit of fold 0 on {N_TRAIN} references; {folder} is fold {receipt.get('fold')}, "
                         f"{receipt['references']} references (only a z-summary.npz fits another run)")
    n_voxels = receipt["voxel_columns"]
    with np.load(archive, allow_pickle=False) as arrays:
        refits = np.unpackbits(arrays["refit_flags_packed"], axis=-1, count=n_voxels, bitorder="little")
        counts, rows = arrays["recurrence_counts"], arrays["held_out_cohort_rows_zero_based"]
    if len(refits) != N_REFITS or not np.array_equal(refits.sum(axis=0, dtype=np.uint8), counts):
        raise ValueError(f"{archive}: {len(refits)} refits whose sum differs from recurrence_counts")
    if original_z is not None:
        original = original_flags(original_z, counts, rows)
    else:
        with np.load(archive, allow_pickle=False) as arrays:
            original = np.unpackbits(arrays["original_flags_packed"], axis=-1, count=n_voxels, bitorder="little").astype(bool)
    ranking = removed_ranking(original, counts, rows, on_planes)
    out = folder / "removed-ranking.tsv"
    table(out, ranking)
    logger.info("n=%d: median removed %.0f voxels, most %d (held-out row %d); wrote %s", receipt.get("reference_n", receipt["references"]),
                np.median([r["removed_voxels"] for r in ranking]), ranking[0]["removed_voxels"],
                ranking[0]["held_out_row_zero_based"], out)
    return out
