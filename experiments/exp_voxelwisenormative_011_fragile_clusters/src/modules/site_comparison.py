"""One cross-fit fold's per-refit export from two sites: flag counts and the fragile-cluster ranking, not bit equality.

The same fold, draws and inputs run on two machines can differ in the last bits of the optimiser, so a voxel near
|z| = 1.96 may flag on one and not on the other. What has to agree is what the slides use: how many flags there are,
which voxels recur in at least MIN_HITS of the 64 refits, and which scans carry the largest unstable clusters.
Individual differing flags are counted for context only. Nothing here writes a file.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from shared.voxelwisenormative.artifacts import sha256

from .clusters import ClusterGrid, bootstrap_ranking
from .inputs import MIN_HITS, THRESHOLD

logger = logging.getLogger(__name__)
MISSING = object()
SAME = ("mode", "held_out_scans", "voxel_columns", "references", "bootstrap_refits", "bootstrap_seed",
        "bootstrap_cohort_rows_sha256", "optimizer_iterations", "flag_rule", "mask_column_order", "packed_axis",
        "packed_bitorder", "reference_cohort_rows_sha256")
DEFAULTS = {"fold": 0, "reference_n": None}  # receipts written before --fold and --reference-n exist lack these keys
TOP = (1, 5, 10, 20)
POPCOUNT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rank correlation (Pearson on average ranks); None when either side is constant."""
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


@dataclass(frozen=True)
class Export:
    """One site's per-refit export, read and checked against its receipt."""

    label: str
    receipt: dict
    packed: np.ndarray  # (refits, scans, ceil(voxels / 8)) uint8, little bit order
    counts: np.ndarray  # (scans, voxels) uint8: in how many refits each voxel was flagged
    rows: np.ndarray  # (scans,) cohort row of each held-out scan


def locate(results: Path, name: str) -> tuple[Path, Path]:
    """(result.json, refit-flags.npz) of a run given as a folder or as a name.

    A name is looked up as `results/<name>`, `results/fetch-<name>` and the archive `results/hugging-face/<name>`.
    """
    for base in (Path(name), results / name, results / f"fetch-{name}", results / "hugging-face" / name):
        if (base / "result.json").is_file() and (base / "refit-flags.npz").is_file():
            return base / "result.json", base / "refit-flags.npz"
        archived = (base / "json" / base.name / "result.json", base / "arrays" / base.name / "refit-flags.npz")
        if all(path.is_file() for path in archived):
            return archived
    raise FileNotFoundError(f"no result.json and refit-flags.npz for {name!r} in {results}")


def refit_sum(packed: np.ndarray, voxels: int) -> np.ndarray:
    """Flags per scan and voxel summed over the refits, unpacked one refit at a time."""
    total = np.zeros((packed.shape[1], voxels), dtype=np.uint8)
    for refit in packed:
        total += np.unpackbits(refit, axis=-1, count=voxels, bitorder="little")
    return total


def check_export(export: Export) -> None:
    """The arrays have the receipt's shapes, and the recurrence counts are the sum of the refits' flags."""
    refits, scans, width = export.packed.shape
    voxels = export.receipt["voxel_columns"]
    if (export.packed.dtype, export.counts.dtype) != (np.uint8, np.uint8) or export.counts.shape != (scans, voxels) \
            or width != (voxels + 7) // 8 or export.rows.shape != (scans,) or refits != export.receipt["bootstrap_refits"]:
        raise ValueError(f"{export.label}: array shapes {export.packed.shape}, {export.counts.shape}, {export.rows.shape} "
                         f"do not match the receipt ({voxels} voxels, {export.receipt['bootstrap_refits']} refits)")
    if not np.array_equal(refit_sum(export.packed, voxels), export.counts):
        raise ValueError(f"{export.label}: recurrence_counts differ from the sum of the refits' flags")


def load(label: str, name: str, results: Path) -> Export:
    """Read one run; its file must be the one its receipt describes (sha256 and size)."""
    receipt_path, archive = locate(results, name)
    receipt = json.loads(receipt_path.read_text())
    record = receipt["outputs"]["refit-flags.npz"]
    if (sha256(archive), archive.stat().st_size) != (record["sha256"], record["bytes"]):
        raise ValueError(f"{archive} is not the export {receipt_path} describes: {record}")
    with np.load(archive, allow_pickle=False) as arrays:
        export = Export(label, receipt, arrays["refit_flags_packed"], arrays["recurrence_counts"],
                        arrays["held_out_cohort_rows_zero_based"])
    check_export(export)
    return export


def require_same_experiment(a: Export, b: Export) -> list[str]:
    """Stop unless both sites ran the same fold, draws and rule; returns the fields only one receipt records."""
    notes = []
    for key in (*SAME, *DEFAULTS):
        first, second = (e.receipt.get(key, DEFAULTS.get(key, MISSING)) for e in (a, b))
        if first is MISSING or second is MISSING:
            if first is not second:
                notes.append(f"{key} is recorded by {a.label if second is MISSING else b.label} only, so it is not compared")
        elif first != second:
            raise ValueError(f"not the same experiment: {key} is {first!r} at {a.label} and {second!r} at {b.label}")
    if not np.array_equal(a.rows, b.rows):
        raise ValueError("not the same experiment: the held-out cohort rows differ")
    return notes


def flag_statistics(a: Export, b: Export) -> dict:
    """Flags per refit and per scan at each site, and how many individual flags differ (context only)."""
    totals = {}
    for e in (a, b):
        bits = POPCOUNT[e.packed]
        totals[e.label] = {"per_refit": bits.sum(axis=(1, 2), dtype=np.int64), "per_scan": bits.sum(axis=(0, 2), dtype=np.int64)}
    differing = POPCOUNT[a.packed ^ b.packed]
    all_a, all_b = totals[a.label]["per_refit"].sum(), totals[b.label]["per_refit"].sum()
    scan_a, scan_b = totals[a.label]["per_scan"], totals[b.label]["per_scan"]
    refit_a, refit_b = totals[a.label]["per_refit"], totals[b.label]["per_refit"]
    return {"flags_all_refits": {a.label: int(all_a), b.label: int(all_b)},
            "relative_difference_all_refits": float((all_b - all_a) / all_a),
            "relative_difference_per_refit_max_abs": float(np.max(np.abs(refit_b - refit_a) / refit_a)),
            "flags_per_refit_range": {e.label: [int(totals[e.label]["per_refit"].min()), int(totals[e.label]["per_refit"].max())] for e in (a, b)},
            "per_scan_flags_spearman": spearman(scan_a, scan_b),
            "per_scan_relative_difference_max_abs": float(np.max(np.abs(scan_b - scan_a) / np.maximum(scan_a, 1))),
            "flags_that_differ": int(differing.sum(dtype=np.int64)), "flag_entries": int(a.packed.shape[0] * a.counts.size)}


def recurrence_statistics(a: Export, b: Export) -> dict:
    """Recurrence counts: how many entries differ and by how much, and who is stable (>= MIN_HITS) or unstable (1 to MIN_HITS - 1)."""
    gap = np.abs(a.counts.astype(np.int16) - b.counts.astype(np.int16))
    stable = {e.label: e.counts >= MIN_HITS for e in (a, b)}
    unstable = {e.label: (e.counts > 0) & (e.counts < MIN_HITS) for e in (a, b)}
    per_scan = {e.label: unstable[e.label].sum(axis=1) for e in (a, b)}
    return {"entries": int(gap.size), "entries_that_differ": int((gap > 0).sum()), "max_gap": int(gap.max()),
            "gap_1": int((gap == 1).sum()), "gap_2": int((gap == 2).sum()), "gap_3_or_more": int((gap >= 3).sum()),
            "stable_voxels": {e.label: int(stable[e.label].sum()) for e in (a, b)},
            "stable_class_changed": int((stable[a.label] != stable[b.label]).sum()),
            "unstable_voxels": {e.label: int(unstable[e.label].sum()) for e in (a, b)},
            "unstable_class_changed": int((unstable[a.label] != unstable[b.label]).sum()),
            "per_scan_unstable_spearman": spearman(per_scan[a.label], per_scan[b.label])}


def ranking_statistics(a: Export, b: Export, grid: ClusterGrid) -> dict:
    """Rank the scans by their largest unstable cluster (refit 1 flags among voxels recurring 1 to 51 times) at each site."""
    ranked = {}
    for e in (a, b):
        first = np.unpackbits(e.packed[0], axis=-1, count=grid.n, bitorder="little").astype(bool)
        ranked[e.label] = bootstrap_ranking(e.counts, first, grid)
    by_row = {label: {r["held_out_row_zero_based"]: r for r in rows} for label, rows in ranked.items()}
    order = sorted(by_row[a.label])
    size = {label: np.array([by_row[label][i]["largest_cluster_voxels"] for i in order]) for label in by_row}
    rank = {label: np.array([by_row[label][i]["rank"] for i in order]) for label in by_row}
    top = {label: [r["held_out_row_zero_based"] for r in rows] for label, rows in ranked.items()}
    shift = np.abs(rank[a.label] - rank[b.label])
    spread = np.abs(size[a.label] - size[b.label])
    return {"largest_cluster_spearman": spearman(size[a.label], size[b.label]),
            "rank_1": {e.label: {"held_out_row": top[e.label][0], "largest_cluster_voxels": int(size[e.label][order.index(top[e.label][0])])}
                       for e in (a, b)},
            "top_k_shared": {str(k): len(set(top[a.label][:k]) & set(top[b.label][:k])) for k in TOP if k <= len(order)},
            "scans_with_a_different_rank": int((shift > 0).sum()), "max_rank_shift": int(shift.max()),
            "largest_cluster_median_abs_difference": float(np.median(spread)), "largest_cluster_max_abs_difference": int(spread.max()),
            "largest_cluster_range": {e.label: [int(size[e.label].min()), int(size[e.label].max())] for e in (a, b)}}


def person_statistics(e: Export, row: int, single: np.ndarray | None, on_planes: np.ndarray) -> dict:
    """What the slides say about one held-out scan; `single` (the one fit's flags, fold 0 only) adds the kept share."""
    counts, refits = e.counts[row], e.packed.shape[0]
    first = np.unpackbits(e.packed[0, row], count=counts.size, bitorder="little").astype(bool)
    out = {"cohort_row": int(e.rows[row]), "refit_1_flags": int(first.sum()), "voxels_in_all_refits": int((counts == refits).sum()),
           "voxels_in_some_not_all": int(((counts > 0) & (counts < refits)).sum()), "voxels_at_least_min_hits": int((counts >= MIN_HITS).sum())}
    if single is not None:
        kept = single & (counts == refits)
        out |= {"single_fit_flags": int(single.sum()), "single_fit_kept_in_all_refits": int(kept.sum()),
                "kept_share": float(kept.sum() / max(int(single.sum()), 1)), "single_fit_stable": int((single & (counts >= MIN_HITS)).sum()),
                "single_fit_fragile": int((single & (counts < MIN_HITS)).sum()),
                "stable_on_planes": int((single & (counts >= MIN_HITS) & on_planes).sum()),
                "fragile_on_planes": int((single & (counts < MIN_HITS) & on_planes).sum())}
    return out


def people(a: Export, b: Export, rows: list[int], single_z: np.ndarray | None, on_planes: np.ndarray) -> list[dict]:
    """person_statistics of the chosen held-out rows at both sites; `single_z` is exp_003's z_test (fold 0 only)."""
    if any(not 0 <= r < a.rows.size for r in rows):
        raise ValueError(f"held-out rows {rows} outside 0..{a.rows.size - 1}")
    if single_z is not None and (a.receipt.get("fold", 0) != 0 or single_z.shape != a.counts.shape):
        raise ValueError(f"the single fit's z exists for fold 0 and its shape {single_z.shape} must be {a.counts.shape}")
    out = []
    for row in rows:
        single = None if single_z is None else np.abs(np.asarray(single_z[row])) > THRESHOLD
        out.append({"held_out_row": row, **{e.label: person_statistics(e, row, single, on_planes) for e in (a, b)}})
    return out


def render_people(table: list[dict], a: str, b: str) -> list[str]:
    """One line per person and number: `a / b`, marked `!` where the two sites differ."""
    names = {"refit_1_flags": "flags in refit 1", "voxels_in_all_refits": "voxels flagged in all refits",
             "voxels_in_some_not_all": "voxels flagged in some but not all", "voxels_at_least_min_hits": f"voxels flagged in {MIN_HITS}+ refits",
             "single_fit_flags": "single-fit flags", "single_fit_kept_in_all_refits": "  of them kept in all refits",
             "kept_share": "  kept share", "single_fit_stable": f"  stable ({MIN_HITS}+ refits)", "single_fit_fragile": "  fragile (fewer)",
             "stable_on_planes": "  stable on the seven planes", "fragile_on_planes": "  fragile on the seven planes"}
    lines = []
    for person in table:
        lines.append(f"held-out row {person['held_out_row']} (cohort row {person[a]['cohort_row']}): {a} / {b}")
        for key, name in names.items():
            if key in person[a]:
                x, y = person[a][key], person[b][key]
                text = f"{x:.1%} / {y:.1%}" if key == "kept_share" else f"{x:,} / {y:,}"
                lines.append(f"  {name}: {text}{'  !' if x != y else ''}")
    return lines


def compare(a: Export, b: Export, grid: ClusterGrid) -> dict:
    """Everything `render` prints: the fields not compared, flag counts, recurrence and the cluster ranking."""
    return {"fold": a.receipt.get("fold", 0), "reference_n": a.receipt.get("reference_n"), "scans": int(a.rows.size),
            "sites": {e.label: e.receipt.get("runtime", {}) for e in (a, b)},
            "not_compared": require_same_experiment(a, b), "flags": flag_statistics(a, b),
            "recurrence": recurrence_statistics(a, b), "ranking": ranking_statistics(a, b, grid)}


def render(report: dict, a: str, b: str) -> list[str]:
    """The report as text lines; `a` and `b` are the two site labels."""
    f, r, k = report["flags"], report["recurrence"], report["ranking"]

    def both(values: dict, fmt: str = "{:,}") -> str:
        return f"{fmt.format(values[a])} / {fmt.format(values[b])}"

    def corr(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.5f}"

    lines = [f"fold {report['fold']}" + (f", reference {report['reference_n']}" if report["reference_n"] else "")
             + f", {report['scans']} held-out scans; {a} / {b}",
             f"  runtime: {a} {report['sites'][a]}", f"  runtime: {b} {report['sites'][b]}"]
    lines += [f"  not compared: {note}" for note in report["not_compared"]]
    lines += ["flags (all 64 refits)",
              f"  total {both(f['flags_all_refits'])}; difference {f['relative_difference_all_refits']:+.4%}; worst single refit {f['relative_difference_per_refit_max_abs']:.4%}",
              f"  per refit {a} {f['flags_per_refit_range'][a][0]:,} to {f['flags_per_refit_range'][a][1]:,}; {b} {f['flags_per_refit_range'][b][0]:,} to {f['flags_per_refit_range'][b][1]:,}",
              f"  per scan: Spearman {corr(f['per_scan_flags_spearman'])}, worst relative difference {f['per_scan_relative_difference_max_abs']:.4%}",
              f"  individual flags that differ (context): {f['flags_that_differ']:,} of {f['flag_entries']:,}",
              f"recurrence counts (out of 64 refits; stable = {MIN_HITS} or more)",
              f"  entries that differ {r['entries_that_differ']:,} of {r['entries']:,}; by 1: {r['gap_1']:,}, by 2: {r['gap_2']:,}, by 3 or more: {r['gap_3_or_more']:,}; largest gap {r['max_gap']}",
              f"  stable voxels {both(r['stable_voxels'])}; changed class {r['stable_class_changed']:,}",
              f"  unstable voxels {both(r['unstable_voxels'])}; changed class {r['unstable_class_changed']:,}; per scan Spearman {corr(r['per_scan_unstable_spearman'])}",
              "fragile-cluster ranking (largest unstable cluster of refit 1, per scan)",
              f"  Spearman of the largest-cluster size {corr(k['largest_cluster_spearman'])}",
              f"  rank 1: {a} row {k['rank_1'][a]['held_out_row']} ({k['rank_1'][a]['largest_cluster_voxels']:,} voxels), {b} row {k['rank_1'][b]['held_out_row']} ({k['rank_1'][b]['largest_cluster_voxels']:,} voxels)",
              "  scans shared by the top k: " + ", ".join(f"{n} of {j}" for j, n in k["top_k_shared"].items()),
              f"  scans with another rank: {k['scans_with_a_different_rank']} of {report['scans']}; largest rank shift {k['max_rank_shift']}",
              f"  largest-cluster size differs by a median of {k['largest_cluster_median_abs_difference']:.1f} voxels, at most {k['largest_cluster_max_abs_difference']:,}"]
    return lines
