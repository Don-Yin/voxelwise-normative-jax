"""Join one A100 sweep and one PCNtoolkit memory run into tables and promote them as a frozen artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

GPU_COLUMNS = ("variant", "dtype", "n_voxels", "seconds_median", "seconds_best", "seconds", "warmup_seconds",
               "peak_bytes_in_use", "peak_pool_bytes", "host_peak_rss_bytes", "z_absdiff_max",
               "reference_voxels_in_subset", "device", "slurm_job_id")
PCN_COLUMNS = ("n_voxels", "peak_rss_bytes", "rss_before_fit_bytes", "rss_after_fit_bytes", "peak_set_during_fit",
               "retained_arrays_lower_bound_bytes", "fit_seconds", "design_columns", "n_train", "n_test", "machine")
GPU_GRID = (1000, 2000, 5000, 10000, 20000, 32768, 50000, 100000, 165402)


def _rows(folder: Path) -> list[dict]:
    """every case JSON of one run, failing on a missing folder."""
    files = sorted(folder.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no case JSON under {folder}")
    return [json.loads(path.read_text()) for path in files]


def _write(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    """write selected columns, joining list values with spaces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: " ".join(map(str, row[key])) if isinstance(row.get(key), list) else row.get(key)
                             for key in columns})


def gpu_table(folder: Path) -> list[dict]:
    """A100 cases ordered by variant and N; both variants must cover the whole grid and pass the gate."""
    rows = []
    for row in _rows(folder):
        row["peak_pool_bytes"] = row["memory_end"].get("peak_pool_bytes")
        rows.append(row)
    for variant in ("early-f32", "baseline"):
        grid = sorted(row["n_voxels"] for row in rows if row["variant"] == variant)
        if tuple(grid) != GPU_GRID:
            raise ValueError(f"{folder}: {variant} covers {grid}, expected {GPU_GRID}")
    if any(not row["gate_passed"] or row["platform"] != "gpu" for row in rows):
        raise ValueError(f"{folder}: a case failed its z gate or did not run on a GPU")
    return sorted(rows, key=lambda row: (row["variant"] != "early-f32", row["n_voxels"]))


def pcntoolkit_table(folder: Path) -> list[dict]:
    """unprofiled PCNtoolkit workers ordered by N."""
    rows = sorted(_rows(folder), key=lambda row: row["n_voxels"])
    if any(row["profiled"] or row["design_columns"] != 12 for row in rows):
        raise ValueError(f"{folder}: a worker was profiled or had a different design")
    return rows


def _sha256(path: Path) -> str:
    """file digest for the artifact manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def promote(gpu_run: Path, pcn_run: Path, results: Path, artifact: Path) -> None:
    """write both tables under .results/tables/ and freeze them, never overwriting an existing artifact."""
    tables = {"gpu-scaling.csv": (gpu_table(gpu_run), GPU_COLUMNS),
              "pcntoolkit-synthetic-memory.csv": (pcntoolkit_table(pcn_run), PCN_COLUMNS)}
    for name, (rows, columns) in tables.items():
        _write(results / "tables" / name, rows, columns)
    if (artifact / "manifest.json").exists():
        raise FileExistsError(f"{artifact} is frozen; create a new version instead")
    artifact.mkdir(parents=True, exist_ok=True)
    for name in tables:
        (artifact / name).write_bytes((results / "tables" / name).read_bytes())
    files = [{"path": "README.md", "storage": "git"}]
    files += [{"path": name, "storage": "git", "sha256": _sha256(artifact / name)} for name in tables]
    manifest = {"schema_version": 1, "artifact_id": artifact.name,
                "producer_experiment": "exp_voxelwisenormative_006_speed_memory_scaling",
                "inputs": {"gpu": str(gpu_run.relative_to(results)), "pcntoolkit": str(pcn_run.relative_to(results))},
                "files": files}
    (artifact / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("promoted %s", artifact)
