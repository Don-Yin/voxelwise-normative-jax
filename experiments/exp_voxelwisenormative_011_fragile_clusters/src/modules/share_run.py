"""Run the fragile-cluster share on the fetched folds: label, measure, aggregate and publish one new run folder."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from shared.voxelwisenormative.artifacts import sha256

from .cluster_share import (
    BANDS,
    DISSOLVED_AT_MOST,
    HALF_REFITS,
    READINGS,
    RECOVERED_ABOVE,
    RESELECTED_ABOVE,
    SIZE_CLASSES,
    summarise,
)
from .inputs import N_REFITS, THRESHOLD, read_mask
from .publish import table
from .share_inputs import FoldRun, check_cohort, open_fold, read_cutoff_fold, read_fold
from .stability import CONNECTIVITIES, fold_columns, grids

logger = logging.getLogger(__name__)
DEFINITIONS = {
    "connectivity": {"main": 26, "sensitivity": [18, 6], "package_defaults": {"FSL cluster": 26, "SPM": 18, "AFNI NN=1": 6},
                      "why_main": "Roels 2016 uses FSL's 26-neighbour clustering"},
    "size_classes_min_voxels": SIZE_CLASSES, "voxel_mm3": 8, "readings": READINGS,
    "jaccard": {"per_refit": "largest Jaccard overlap between the cluster and any component of that refit's flag map",
                "bands_of_the_mean": f"fpc manual p. 44, not the paper: below {BANDS[0]} not to be trusted, {BANDS[0]} to {BANDS[1]} doubtful, "
                                     f"{BANDS[1]} to {BANDS[2]} stable, from {BANDS[2]} highly stable",
                "per_refit_counts": f"Hennig 2007 p. 7: dissolution = best Jaccard <= {DISSOLVED_AT_MOST}, recovery = best Jaccard > {RECOVERED_ABOVE}; "
                                    "no cut-off on how many refits, so only the distribution is reported"},
    "reselection": f"Roels 2016: mean over the cluster's voxels of recurrence_counts / {N_REFITS}; reselected above {RESELECTED_ABOVE[0]}/{RESELECTED_ABOVE[1]}",
    "refits_with_fewer_than_half_flagged": HALF_REFITS,
    "per_person": "over the people with at least one cluster of the class; quartiles by linear interpolation",
    "x_of_5": "5 times the share", "size_bins": "doubling bins of voxels: 1, 2, 3-4, 5-8, 9-16, ...",
}


def definitions(threshold: float) -> dict:
    """The definitions as used at this run's cutoff, as written into result.json."""
    return {"cluster": f"connected component of the single fit's map, positive (z > {threshold:g}) and negative (z < -{threshold:g}) separately",
            "refit_flags": f"unsigned: abs(z) > {threshold:g} in each of the {N_REFITS} refits", **DEFINITIONS}


def table_rows(result: dict) -> tuple[list[dict], list[dict]]:
    """Flat rows for the two TSV tables: shares by reading and size class, and the size curve."""
    by_reading, by_size = [], []
    for neighbours in CONNECTIVITIES:
        block = result[f"connectivity_{neighbours}"]
        for scope, data in block["scopes"].items():
            for reading, reading_data in data["readings"].items():
                for cls, s in reading_data["size_classes"].items():
                    by_reading.append({"connectivity": neighbours, "scope": scope, "people": data["people"], "reading": reading, "size_class": cls,
                                       "clusters": s["clusters"], "fragile": s["fragile"], "pooled_share": s["pooled_share"],
                                       "pooled_x_of_5": s["pooled_x_of_5"], "people_with_clusters": s["people_with_clusters"],
                                       **{f"person_{k}": (s["per_person_share"] or {}).get(k) for k in ("q1", "median", "q3")},
                                       "fragile_volume_share": s["fragile_volume_share"]})
        curve = block["size_curve"]
        for scope, by_reading_curve in curve["scopes"].items():
            for reading, c in by_reading_curve.items():
                for i, (low, high) in enumerate(zip(curve["size_from"], curve["size_to"], strict=True)):
                    by_size.append({"connectivity": neighbours, "scope": scope, "reading": reading, "size_from": low, "size_to": high,
                                    "clusters": c["clusters"][i], "fragile": c["fragile"][i], "share": c["share"][i]})
    return by_reading, by_size


def record(path: Path, results: Path) -> dict:
    """Where an output file is (relative to .results), its SHA-256 and its size."""
    return {"path": str(path.relative_to(results)), "sha256": sha256(path), "bytes": path.stat().st_size}


def publish(results: Path, name: str, columns: dict[int, dict[str, Any]], result: dict) -> None:
    """Write the per-cluster columns, the aggregate JSON and two TSV tables into new folders; the JSON goes last."""
    folders = {kind: results / kind / name for kind in ("arrays", "json", "tables")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for neighbours, data in columns.items():
        path = folders["arrays"] / f"clusters-{neighbours}.npz"
        np.savez_compressed(path, **data)
        outputs[path.name] = record(path, results)
    by_reading, by_size = table_rows(result)
    for file, rows in (("share-by-reading.tsv", by_reading), ("share-by-size.tsv", by_size)):
        path = folders["tables"] / file
        table(path, rows)
        outputs[file] = record(path, results)
    (folders["json"] / "result.json").write_text(json.dumps({**result, "outputs": outputs}, indent=2, allow_nan=False) + "\n")


def run(results: Path, name: str, folds: list[int], mask_path: Path, threshold: float = THRESHOLD) -> None:
    """Measure the clusters of the given folds (fold 0 first) at `abs(z) > threshold` and publish their shares as run `name`.

    1.96 reads the z-summary runs; a stricter cutoff reads the cutoff-flags runs, which hold each refit's flags at it.
    """
    started = time.perf_counter()
    if any((results / kind / name).exists() for kind in ("arrays", "json", "tables")):
        raise FileExistsError(f"refusing to overwrite run {name} under {results}")
    mask, _, _ = read_mask(mask_path)
    cutoff = threshold != THRESHOLD
    runs: list[FoldRun] = [open_fold(results, k, cutoffs=cutoff) for k in folds]
    grid_by_connectivity = grids(mask)
    parts: dict[int, list[dict[str, np.ndarray]]] = {neighbours: [] for neighbours in CONNECTIVITIES}
    rows_by_fold = {}
    for fold_run in runs:
        arrays = read_cutoff_fold(fold_run, threshold) if cutoff else read_fold(fold_run)
        rows_by_fold[fold_run.fold] = arrays.rows
        for neighbours, data in fold_columns(grid_by_connectivity, arrays.z, arrays.packed, f"fold {fold_run.fold}", threshold).items():
            data["fold"] = np.full(len(data["voxels"]), fold_run.fold, dtype=np.int8)
            data["cohort_row"] = arrays.rows[data["person"]]
            parts[neighbours].append(data)
    check_cohort(rows_by_fold)
    columns = {neighbours: {key: np.concatenate([p[key] for p in ps]) for key in ps[0]} for neighbours, ps in parts.items()}
    result = {"schema_version": 1, "run_name": name, "folds": folds, "refits": N_REFITS, "threshold": threshold, "definitions": definitions(threshold),
              "inputs": {f"fold{r.fold}": {"run": r.name, "hugging_face_job": r.receipt["runtime"]["hugging_face_job"],
                                          "references": r.receipt["references"], "held_out_scans": r.receipt["held_out_scans"],
                                          "files": r.files} for r in runs}}
    for neighbours, data in columns.items():
        result[f"connectivity_{neighbours}"] = summarise(data, folds)
    result["seconds"] = time.perf_counter() - started
    publish(results, name, columns, result)
    logger.info("run %s: %d people, clusters %s; %.1f s", name, sum(len(v) for v in rows_by_fold.values()),
                ", ".join(f"{len(columns[n]['voxels']):,} at {n} neighbours" for n in CONNECTIVITIES), result["seconds"])
