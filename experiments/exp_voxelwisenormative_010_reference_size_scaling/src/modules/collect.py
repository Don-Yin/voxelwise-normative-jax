"""Join only complete, explicit campaigns; freeze the inputs the slide actually uses."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from shared.voxelwisenormative.artifacts import resolve
from shared.voxelwisenormative.receipts import publish
from shared.voxelwisenormative.reference_grid import (
    BENCHMARK,
    BOOT_CASES,
    CPU_CASES,
    GPU_CASES,
    MASK_VOXELS,
    MIN_RECUR,
    N_HELD_OUT,
    PRECISIONS,
    REAL,
    TRIALS,
    V_SWEEP,
    Z_THRESHOLD,
)

from .scaling import cpu_projection, estimate_cost, n_powers, power_fit

PRIOR_SWEEP = "voxelwise-speed-memory-scaling-v1"


def read_json(path: Path) -> dict:
    """Require a complete JSON object, naming the source on any failure."""
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read receipt {path}: {error}") from error
    if not isinstance(row, dict):
        raise TypeError(f"expected a JSON object in {path}, got {type(row).__name__}")
    return row


def read_grid(results: Path, stage: str, runs: list[str], expected: dict) -> dict:
    """Load exactly one receipt for every wanted case, with no implicit latest-run selection."""
    found = {}
    for run in runs:
        root = results / "json" / stage / run
        files = sorted(root.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"no {stage} case receipts in {root}")
        for path in files:
            if path.name in found or path.name not in expected:
                raise ValueError(f"duplicate or unexpected {stage} case: {path}")
            row = read_json(path)
            required = {"schema_version": 1, "n_test": N_HELD_OUT, **expected[path.name]}
            if run.isdecimal():
                required["slurm_array_job_id"] = run
            for key, value in required.items():
                if row.get(key) != value:
                    raise ValueError(f"{path}: got {key}={row.get(key)!r}, expected {value!r}")
            row["receipt_path"] = str(path.relative_to(results))
            found[path.name] = row
    if missing := sorted(set(expected) - set(found)):
        raise ValueError(f"missing {stage} cases: {missing}")
    return found


def positive(row: dict, *fields: str) -> None:
    """Require finite positive observations before logs and scale factors."""
    for key in fields:
        v = row.get(key)
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{row['receipt_path']}: expected positive {key}, got {v!r}")


def previous_voxel_sweep() -> dict:
    """Reuse the four requested mask widths from exp_006's frozen A100 table."""
    source = resolve(PRIOR_SWEEP, "gpu-scaling.csv")["gpu-scaling.csv"]
    columns = ("n_voxels", "seconds_median", "peak_bytes_in_use", "host_peak_rss_bytes")
    with source.open(newline="", encoding="utf-8") as handle:
        rows = [dict(found, **{key: float(found[key]) if key == "seconds_median" else int(found[key])
                              for key in columns})
                for found in csv.DictReader(handle) if int(found["n_voxels"]) in V_SWEEP]
    powers = {}
    for variant in ("early-f32", "baseline"):
        series = [r for r in rows if r["variant"] == variant]
        if len(series) != len(V_SWEEP) or {r["n_voxels"] for r in series} != set(V_SWEEP):
            raise ValueError(f"incomplete {variant} voxel sweep in {source}")
        powers[variant] = {key: power_fit((int(r["n_voxels"]), float(r[key])) for r in series)
                           for key in ("seconds_median", "peak_bytes_in_use")}
    return {"artifact_id": PRIOR_SWEEP, "file": source.name, "n": 450, "rows": rows, "powers": powers}


def collect(results: Path, name: str, cpu_runs: list[str], gpu_runs: list[str], bootstrap_runs: list[str]) -> Path:
    """Validate the fixed grid and publish one portable aggregate for the figure and cost CLI."""
    cpu = read_grid(results, "cpu", cpu_runs, {
        f"{cell.label}-v{width:03d}.json": {"workload": "pcntoolkit", "n": cell.n, "dataset": cell.dataset,
                                          "columns": width, "mask_voxels": MASK_VOXELS, "design_columns": 12}
        for cell, width in CPU_CASES})
    gpu = read_grid(results, "gpu", gpu_runs, {
        f"{cell.label}-{precision}.json": {"workload": "batched-early", "n": cell.n, "dataset": cell.dataset,
                                           "columns": MASK_VOXELS, "precision": precision}
        for cell, precision in GPU_CASES})
    boot = read_grid(results, "bootstrap", bootstrap_runs, {
        f"{cell.label}-chain{chain}.json": {"workload": "bootstrap", "n": cell.n, "dataset": "ixi", "chain": chain,
                                           "columns": MASK_VOXELS, "precision": "float64", "trials": TRIALS,
                                           "min_recurring_trials": MIN_RECUR, "z_abs_threshold": Z_THRESHOLD}
        for cell, chain in BOOT_CASES})
    for row in cpu.values():
        positive(row, "seconds", "peak_rss_bytes", "retained_training_arrays_bytes")
    for row in gpu.values():
        positive(row, "seconds_median", "warmup_seconds", "peak_bytes_in_use", "host_peak_rss_bytes")
    for row in boot.values():
        positive(row, "baseline_flags")
        if not (0 <= row["unstable_flags"] <= row["baseline_flags"]
                and row["unstable_fraction"] == row["unstable_flags"] / row["baseline_flags"]):
            raise ValueError(f"{row['receipt_path']}: invalid instability numerator or fraction")
    timings = [
        {"n": c.n, "dataset": c.dataset, "n_test": N_HELD_OUT,
         "pcntoolkit": cpu_projection([cpu[f"{c.label}-v{w:03d}.json"] for w in c.cpu_widths]),
         "gpu": {p: gpu[f"{c.label}-{p}.json"] for p in PRECISIONS}}
        for c in BENCHMARK
    ]
    instability = []
    for cell in REAL:
        chains = [boot[f"{c.label}-chain{k}.json"] for c, k in BOOT_CASES if c == cell]
        fractions = [r["unstable_fraction"] for r in chains]
        instability.append({"n": cell.n, "chain_count": len(chains), "mean": statistics.mean(fractions),
                            "range": [min(fractions), max(fractions)], "chains": chains})
    sweep = previous_voxel_sweep()
    before = next(r for r in sweep["rows"] if r["variant"] == "early-f32" and r["n_voxels"] == MASK_VOXELS)
    now = next(r["gpu"]["float32"] for r in timings if r["dataset"] == "ixi" and r["n"] == 450)
    return publish(results / "json" / "collect" / name / "summary.json",
                   {"schema_version": 1, "runs": {"cpu": cpu_runs, "gpu": gpu_runs, "bootstrap": bootstrap_runs},
                    "benchmarks": timings, "instability": instability, "n_powers": n_powers(timings),
                    "voxel_sweep": sweep, "ukb_scenario_50000": estimate_cost(timings, 50_000),
                    "exp006_comparison_at_450": {"warm_time_ratio": now["seconds_median"] / before["seconds_median"],
                                                 "live_bytes_ratio": now["peak_bytes_in_use"] / before["peak_bytes_in_use"],
                                                 "old": before}})


def save_cost(results: Path, name: str, n: int) -> Path:
    """Evaluate a new requested size against a particular collected campaign."""
    source = results / "json" / "collect" / name / "summary.json"
    summary = read_json(source)
    return publish(results / "json" / "cost" / name / f"n{n:05d}.json",
                   {"source": str(source.relative_to(results)), **estimate_cost(summary["benchmarks"], n)})
